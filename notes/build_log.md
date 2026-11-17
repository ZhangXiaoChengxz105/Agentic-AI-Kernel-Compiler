# Build log: Tasks 1.2–1.8 and 2.1–2.4 (2026-11-15)

Everything below runs on the login node (CPU) unless marked **GPU**. Nothing was submitted to Slurm,
no Lumen chat call was made (only the free `/v1/models` and `/v1/usage`), and nothing was pushed.
Earlier part (first VOLTA loop on softmax): `notes/volta_loop_log.md`.

## 0. Repository map
```
kernels/common.cuh               EXP() macro: __symexpf for VOLTA builds, expf for GPU builds
kernels/<op>/{<op>_ref,<op>_opt}.cu, bugs/*.cu, manifest.json     op ∈ softmax, gemm, layernorm
verify/   ptx.py (B1 compile) · volta.py (B2 run+parse VOLTA) · cases.py (B3 per-op VOLTA layouts)
ref/      inputs.py (value distributions) · ops.py (B5 golden refs + per-op launch/args)        Task 1.4
harness/  testcase.py · check.py (B6) · launch.py (B7) · runner.py/worker.py (B8) · fixed_suite.py
          bench.py (B9) · cli.py (check / bench)                                             Task 1.5/1.6
scripts/slurm/  gpu_job.sbatch · smoke.sbatch · interactive.sh (B10)                         Task 1.7
agent/lumen.py  (B11)                                                                        Task 1.8
changeaware/    diff.py (B12) · classify.py (B13) · generate.py (B14) · __init__.plan()      Task 2
scripts/run_volta.py (B4) · scripts/run_change_aware.py (B15)
tests/    test_volta.py · test_harness.py · test_lumen.py · test_change_aware.py  (64 tests, 3 GPU-only)
results/  volta/*.json · change_aware/summary_*.json · change_aware/tests/<bug>.json
```
Run all tests: `source env.sh && envs/tirx/bin/python -m unittest discover -s tests` → 64 OK, 3 skipped (GPU).

## 1. Kernels and injected bugs (inputs to everything else)
Contract: `extern "C" __global__`, exp written as `EXP(x)`, compile with `-I kernels/`. One CTA per row
(softmax/layernorm, block 128; ref block 1) or per 16×16 output tile (gemm, block 16×16).

| Op | Variant | Category | Single change vs opt |
|---|---|---|---|
| softmax | bug_nosync | sync | barrier removed in max reduction |
| softmax | bug_tail | tiling | reductions stop at (N/128)·128 |
| softmax | bug_offbyone | indexing | sum loop `i <= N` |
| softmax | bug_nomax | numerical | no max subtraction |
| gemm | bug_nosync | sync | barrier after inner product removed (WAR race) |
| gemm | bug_ktail | tiling | ntiles = K/16 instead of ceil |
| gemm | bug_guard | indexing | output guard `<=` (writes past the edge) |
| layernorm | bug_nosync | sync | barrier between reading mean and reusing `red` removed |
| layernorm | bug_tail | tiling | reductions stop at (N/128)·128 |
| layernorm | bug_offbyone | indexing | variance loop `i <= N` |
| layernorm | bug_onepass | numerical | variance = E[x²] − mean² |

`manifest.json` per op: variants (file, role, block, bug{category, change}), `volta_shapes`, and
machine-readable `expect_volta` (`"RACE"`, a list = any-of, or rules `[{"if": "K % 16 == 0", "then": "EQUIV"}, {"then": "NOT_EQUIV"}]`).

## 2. Building blocks (input → output, conditions)

### B1 `verify.ptx.compile_to_ptx(src, out_dir, mode, arch="sm_80")` → `PtxBuild`
- mode `volta`: `-DVOLTA`, no `-lineinfo` (VOLTA drops labels followed by `.loc`). mode `gpu`: `-lineinfo`.
- output: `.ptx` path, entry names, exact nvcc command, seconds; `CompileError(stderr)` on failure. ~0.6 s/file.

### B2 `verify.volta` → `VoltaCase` → `run_compare()` → `VoltaResult`
- input: 2 PTX + kernel names, ref/cand block, grid, arrays (auto-placed), params in order, check arrays, timeout.
- output: `status` ∈ EQUIV, NOT_EQUIV, RACE, DEADLOCK, OUT_OF_BOUNDS, UNINIT, FOOTPRINT, UNSUPPORTED,
  CONFIG_ERROR, TIMEOUT, ERROR; `which` kernel; detail; mismatch count + examples; timings; raw output.
- conditions: EQUIV = equal over ℝ, CTA 0 only, at exactly these sizes.

### B3 `verify.cases.make_case(op, ref_ptx, ref, cand_ptx, cand, shape)` → `VoltaCase`
| op | arrays (kind, length) | params | grid |
|---|---|---|---|
| softmax | x in 2N, y out 2N | x, y, N | (2,) |
| layernorm | x in 2N, w in N, b in N, y out 2N | x, w, b, y, N, eps=1e-5 | (2,) |
| gemm | A in M·K, B in K·N, C out M·N | A, B, C, M, N, K | (⌈N/16⌉, ⌈M/16⌉) |
Base addresses: 64 KiB-aligned with a 64 KiB gap (never overlap). `expected_statuses(expect, shape)` resolves manifest rules.

### B4 `scripts/run_volta.py --op <op> [--shapes "N=256;N=200"] [--only a,b] [--timeout s]`
compile all variants → VOLTA ref vs each candidate per shape → compare with `expect_volta` → table +
`results/volta/<op>_<ts>.json`; exit 0 iff all as expected.

### B5 `ref.ops.OPS[op]` (Task 1.4)
| field | meaning |
|---|---|
| `make_inputs(shape, dist, seed)` | float32 CPU tensors, deterministic (seeded generator) |
| `golden(inputs)` | float64 PyTorch result (the standard answer) |
| `kernel_args(dev_inputs, out, shape)` | `[(value, ctypes type)]` in signature order |
| `launch(shape, block)` | (grid, block) |
| `atol, rtol, magnitude` | fp32 tolerance; gemm: rtol × (|A|@|B|), grows with K like fp error |
| `flops, bytes, torch_impl` | for the benchmark |
Distributions (`ref.inputs`): normal, uniform, large (×100 → exp overflow), neg_large (−200 → exp underflow),
offset (1e4 + N(0,1) → cancellation), tiny (×1e-39 → subnormals), mixed (10^±6), constant.

### B6 `harness.check.compare(buf, out, golden, atol, rtol, scale)` → `CheckResult` (Task 1.5)
- output buffer pre-filled with sentinel NaN `0x7FC0DEAD` + 256 canary elements on each side.
- reason priority: `canary` (out-of-bounds store) > `unwritten` (sentinel left) > `nonfinite` > `mismatch` > ok.
- result: passed, reason, counts, max abs/rel error, up to 5 (index, got, want) examples, message.

### B7 `harness.launch.CudaModule(ptx).get(name)(grid, block, args)` **GPU**
driver API via `cuda-bindings`: `cuModuleLoadData` (PTX JIT, cached in `$CUDA_CACHE_PATH`) + `cuLaunchKernel`
on torch's current stream; raises `CudaError(name, description)`.

### B8 `harness.runner.run_tests[_isolated](op, ptx, kernel, block, tests)` → `[dict]` **GPU**
per test: inputs → golden (fp64 on GPU) → launch `repeat` times → compare first run → bitwise-compare
repeats (`nondeterministic` = race symptom). Isolated version runs each kernel in a child process
(`python -m harness.worker`): a CUDA fault kills only the child; remaining tests become `skipped`.

### B9 `harness.bench.bench(fn, warmup=10, iters=50)` → median/p10/p90/min ms **GPU**
CUDA events per launch, 256 MiB L2 flush before each timed launch. `python -m harness.cli bench --op gemm`
checks correctness on the bench shape first, then times each passing variant and PyTorch.

### B10 Slurm (Task 1.7), **ask before submitting**
`scripts/slurm/gpu_job.sbatch <cmd...>` (1×A100, 30 min, `gpuA100x4-interactive`, logs/slurm/),
`smoke.sbatch` (nvidia-smi + driver, torch, GPU unit tests, fixed suite ×3 repeats on all variants, benchmarks),
`interactive.sh` (asks y/N before `srun --pty`). Validated with `sbatch --test-only` only.

### B11 `agent.lumen.LumenClient(model="glm-5.3")` (Task 1.8)
- free: `models()`, `usage()`, `price(model)`.
- `chat(messages, max_tokens, dry_run, tag)` → `ChatResult(content, reasoning, input/output tokens, cost, latency)`.
- guard: worst case = est_in·price_in + max_tokens·price_out; refuse if > 0.5 coins/call or if it would leave < 1 coin.
- retries 429/5xx ×3; `ConsentRequired` on 403 consent_required; every call appended to `logs/lumen/calls.jsonl`.
- CLI: `python -m agent.lumen models|usage|chat [--dry-run] "prompt"`.

### B12 `changeaware.diff.diff_sources(old, new)` → `[Hunk(kind, old[(line,text)], new[...])]` (Task 2.1)
normalizes first: comments stripped, whitespace collapsed, blank lines dropped, `__global__` names → `KERNEL`.

### B13 `changeaware.classify.classify(hunks)` → `[Label(category, rule, line, evidence)]` (Task 2.2)
categories: tiling, indexing, memory, sync, numerical. Added/deleted lines: line rules. Modified lines:
paired by similarity; construct rules fire if either side mentions it; math rule only if a math call's
text changed; pair rules: comparison op, ±1, loop bound, guard, index expression. `categories()` ranks by evidence.

### B14 `changeaware.generate` (Task 2.3)
- `generate_tests(op, cats, block, budget=16)` → `[TestCase]`, round-robin across categories, deterministic seeds,
  tags explain why ("tiling: N=129 vs block 128").
- `volta_shapes(op, cats, block)` → default shape + shapes with the boundary inside CTA 0
  (row ops N = 200 and 100; gemm K = 40, and M = N = 12). Nothing extra for `numerical` (VOLTA is blind there).

### B15 `scripts/run_change_aware.py`
for every opt→bug pair: plan → VOLTA at the fixed default shape (config B) vs change-aware shapes (config C) →
table + `results/change_aware/summary_<ts>.json` + GPU test lists `results/change_aware/tests/<bug>.json`.

## 3. Results
### 3.1 VOLTA feasibility (Task 1.2): all three ops supported; 32/32 expectations met
| op | shapes | outcome |
|---|---|---|
| softmax | N=256, 200 (+1024, 1000) | opt EQUIV ≤0.5 s; nosync RACE; tail hidden at N÷128, caught otherwise; offbyone NOT_EQUIV; nomax EQUIV (blind spot, 1.3 s → 210 s at N=1024) |
| gemm | (32,32,48), (32,32,40), (12,12,40) | opt EQUIV ≤0.15 s; nosync RACE; ktail caught only for K%16≠0; guard caught only at 12×12 (as a RACE on C: thread (r, N) and (r+1, 0) both write C[(r+1)N]) |
| layernorm | N=128, 100 | opt EQUIV 1.8 s / 0.6 s; nosync RACE; tail caught at N=100; offbyone NOT_EQUIV; onepass EQUIV (blind spot) |

### 3.2 Task 2 on the 11 injected bugs (CPU, VOLTA only)
| | result |
|---|---|
| ground-truth category detected | 11/11 (and ranked first in 11/11) |
| VOLTA detection, configuration B (fixed default shape) | **5/11** |
| VOLTA detection, configuration C (change-aware shapes) | **9/11** (+ softmax/layernorm/gemm tail-type bugs and gemm guard) |
| remaining 2 (numerical: softmax nomax, layernorm onepass) | invisible to VOLTA by design; Task 2 generates `large` / `offset` inputs for them; CPU float32 emulation confirms the harness flags them (nonfinite / mismatch) |
Caveat: the classifier rules and the bugs were written by the same person (me), so 11/11 is an upper bound;
it must be re-measured on agent-written revisions and bugs written by someone else (Task 5).

## 4. Problems and fixes (this session, after the first VOLTA loop)
| # | Problem | Cause | Fix |
|---|---|---|---|
| 7 | old softmax-only driver/cases didn't fit gemm (2-D blocks, 3 shape params) | op-specific code | generic `make_case` + per-op layouts, auto address placement, rule-based expectations |
| 8 | classifier labelled bug_tail "numerical" (changed line contains `fmaxf`) | line-level matching | modified lines paired; numerical only if a math call's text changes |
| 9 | torch install bumped NVRTC 13.2.78 → 13.2.86, conflicting with tirx-harness's pin | exact pins on both sides | kept 13.2.86, re-verified TIRx compile + racecheck |
| 10 | GPU batch queue ~13 days | partition full (2,730 pending) | scripts use `gpuA100x4-interactive` (minutes, ≤1 h) |
| 11 | VOLTA LayerNorm N≥512: >3.5 min, ~10 GB RAM on the login node | O(N²) expansion of the variance | stopped it; VOLTA layernorm shapes ≤ 200; scaling runs belong on a compute node |
| 12 | `pkill -f <pattern>` returned 144 | pattern matched its own shell | harmless; use the PID next time |

## 5. Waiting for approval (update: item 1 done, job 24032011, see notes/gpu_results.md; item 2 done, see notes/glm_pilot.md)
1. **GPU smoke test** (`sbatch scripts/slurm/smoke.sbatch`): 1 A100, ≤30 min on `gpuA100x4-interactive`
   (≈0.5 of the 749 remaining GPU-hours): driver version, 3 GPU unit tests, fixed suite on all 15 variants,
   benchmarks. Then the change-aware GPU tests for the 11 bugs (`results/change_aware/tests/*.json`).
2. **Lumen**: acknowledge `glm-5.3` in the web UI, then one tiny chat call (< 0.01 coins) to verify end to end.
3. **git push** of the local commits.

## 6. Restructure into the akc project + validator interface (2026-11-16)
Why: organize the code as one engineering project around the agent's parts (task, llm, agent, backend,
validation, feedback, evaluation) instead of by task; give all validators one input/output contract so the
loop, the feedback and the evaluation understand a single format. Old→new paths: `notes/README.md`.
What changed:
- `akc/validation/interface.py`: Candidate, ValidationContext, Finding (17 kinds, 3 severities), ValidationReport
  (`passed`, `decided` → coverage, `seconds` → overhead), Validator protocol; parent = last ACCEPTED candidate.
- Validators: FixedTestsValidator (baseline), VoltaValidator (mechanism 1), ChangeAwareValidator (mechanism 2);
  shared executors in `validation/runners/` (GPU for CUDA and tirx, NumSim for tirx on CPU, capped).
- `validation/pipeline.py`: configurations A/B/C; in C the change-aware VOLTA shapes are handed to VOLTA.
- `feedback/render.py`: findings → concise text (errors with example/related change/hint, then "not verified").
- Change-aware is now a qualified validator: manifest-driven (no per-op constants in code), tile sizes read from
  the candidate (`#define TILE 32`, `BLOCK = 256`, `LaunchConfig(block=…)`), misaligned-pointer tests (memory),
  grid sweeps (sync), deep VOLTA shape, `akc hottest` one-shot report.
- Workloads: `workloads/<op>/manifest.json` holds shape keys, launch, tiles, fixed suite, numerical dists, VOLTA
  default/deep/cap, tirx contract, references; softmax manifest includes the tirx variants.
- `akc` CLI (validate, hottest, check, bench, volta, eval-bugs, tirx-gpu, pilot, lumen); `pyproject.toml`
  (`pip install -e .`); `setup_env.sh` + `setup/requirements-akc.lock` (exact versions); docs/architecture.md, docs/setup.md.
Checks: 86 tests (71 migrated + 15 new) pass on the login node (3 GPU + 1 slow NumSim test skipped by default; the
NumSim test passes with AKC_SLOW=1). `akc hottest` on GLM's accepted-but-buggy kernel (A_r3 vs last accepted A_r1)
reports VOLTA NOT_EQUIV at the deep shape N=584 on the CPU alone. `akc eval-bugs --device cpu`: B 5/11, C 9/11
(VOLTA parts; dynamic tests need a GPU).

## 7. Speed as the agent's objective: `akc/perf/` (2026-11-16)
Why: VOLTA and the tests only decide correctness; the agent also needs to know how fast a passing kernel is and
how far it is from what the A100 allows, otherwise "optimize" has no signal. Speed is not a validator: no
pass/fail, measured only for candidates that passed, identical in configurations A/B/C.
| block | input | output | conditions |
|---|---|---|---|
| `perf/hardware.py` | torch device name | `GpuSpec` (DRAM GB/s, FP32 TFLOP/s, ridge) | A100 datasheet values; unknown GPU → A100-40GB numbers, `known=False` |
| `perf/measure.py` | Candidate, shape (default manifest `perf.shape`) | `PerfReport` | GPU only (else `measured=False`); child process; correctness at that shape first; kernel / PyTorch / device-copy timed with `backend/bench.py` |
| `perf/report.py` `analyze` | timings, shape, GpuSpec | roofline numbers | effective bandwidth counts the minimum traffic (read inputs once, write output once) |
| `feedback/render.py` `render_perf` | PerfReport (+ best so far) | text for the agent | memory-bound → % of peak bandwidth + copy floor; compute-bound → % of FP32 peak + PyTorch's TFLOP/s |
Refactors: `runners.build_for_gpu` (shared CUDA→PTX build), `runners.gpu.make_launcher` (shared launch binding);
`akc bench` now uses `perf.measure` (table + `--feedback`, `--lang tirx`, results in `results/perf/`);
`akc validate --perf` prints the full observation (validation, then speed).
Checked on the CPU: the TF32-tensor-core hint is real: rounding GEMM inputs to TF32 gives a relative error of
1.3e-4 vs |A||B| (tolerance 1e-5), so the feedback tells the model full fp32 accuracy is required.
Tests: 101 (15 new in `tests/perf/`), all pass on the login node; 6 skipped (5 GPU, 1 slow NumSim).
GPU run 24036388 (notes/gpu_results.md, second part) confirmed the perf path and found three problems, fixed:
the LayerNorm "offset" false alarm (condition-aware tolerance `_ln_magnitude` in `task/reference.py`; CPU emulation
in `tests/validation/test_runners.py`), the tirx `eval-bugs` crash (`bugset.parent_variant`, `tests/evaluation/`),
and duplicate related-change lines. Tests: 105.

## 8. Classifier robustness probe (2026-11-16)
Script: `scripts/experiments/classifier_probe.py` (CPU, seconds). 13 hand-made semantic edits of `softmax_opt.cu`
and of GLM's `softmax_glm_r3.cu`, plus 2 no-op edits, each labeled by `changeaware.plan`.

| result | edits |
|---|---|
| right category (5/13) | `red[0]`→`red[1]` (indexing), removed `__syncthreads` (sync), `-INFINITY`→`0` (numerical), `BLOCK` 128→256 (tiling), loop start `tid`→`tid+1` (indexing) |
| wrong category (2/13) | `+=`→`=` in the sum reduction → indexing; `__shfl_xor_sync`→`__shfl_down_sync` → numerical |
| no category at all (6/13) | reduction stride `s>>=1`→`s>>=2`; `*inv`→`*inv*inv`; `BLOCK/2`→`BLOCK/4`; GLM's real dropped-term bug `(e2+e3)`→`(e2+l)`; shuffle offset 16→8; unroll guard `> 384`→`> 383` |
| no-op edits (2/2 spurious labels) | renaming `m` → numerical; reformatting → indexing (only costs extra tests) |

Why: the rules are regexes over changed lines, and their vocabulary was written together with our 11 injected bugs
(so 11/11 was an upper bound). Missing vocabulary: warp shuffles, shift/stride updates, numeric constants in loop
conditions, plain arithmetic outside math calls.
Why it has not hurt so far: configuration C always runs the fixed tests, and an edit with no category gets a
fallback set (N=1000, normal + one extreme distribution). By reasoning over the code, all 6 misses would still be
caught by one of those. That is partly luck: the `> 383` guard bug fires only for N = 384 + tid + 512k (tid < 128).
The fixed sizes (256/1024/4096) never hit that; the fallback's N=1000 happens to.
Real GLM revisions (6 consecutive pairs) are mostly large rewrites (11–53 changed lines) and get 2–4 categories.

### 8b. Classifier revision + mutation benchmark (2026-11-16, same day)
After the probe above, the classifier was revised and measured on mechanically generated edits (`akc mutants`,
`akc/evaluation/mutants.py`).

What changed:
- **diff.** Reformatting and consistent renaming of local names are "no change" (`cosmetic_only`).
- **classify.** Loop / `if` headers are split off each line:
  - condition changed → loop-bound;
  - start or step changed → loop-step;
  - if-condition changed → guard.

  The remaining statement, compared with subscripts removed, is numerical "arith" if it assigns a float and
  indexing "index-arith" otherwise. This is a fallback, only used when no other rule fired.
- **new rules.** while-loop bounds; warp shuffles (only when the shuffle call itself changed); any `#define N <int>`
  (compile-constant); SMEM constants (memory); tirx math functions passed as values (`block_reduce(m, txl.max)`).
- **generate.** Floor tests (sizes 1, T±1, 1000, plus one extreme distribution) run on every revision, on top of
  the budget. This replaces the old "no category → 2 sanity tests".
- **plan.** A tirx source that does not parse no longer crashes: every category is assumed.
- **optional LLM second opinion** (`llm_classify.py`): union with the rules, plus up to 4 risky sizes, cached.
  Tested only with a fake client.

Mutation operators: cmp, idx1, const, define, nosync, shfl, shfl_off, aop (float arithmetic only), math, init;
added after the revision (held out): negate, smem, dtype; no-op: rename, format.

Bugs found in the generator along the way, and fixed:
- format split comments into code;
- aop hit index/pointer arithmetic and subscripts;
- the tirx init mutant used CUDA's `0.0f`.

| kernels | before | after |
|---|---|---|
| tuning set (3 ours + 4 GLM CUDA + tirx ref) | 132/198 right, 60 none, no-op 25/31 labeled | 198/198, 0, 0/31 |
| held out (4 GLM CUDA + 4 GLM tirx, never looked at) | 117/189, 70 none, 16/28 | 189/189, 0, 0/28 |

The 13 hand-made probe edits of section 8 are now all classified as expected, including GLM's dropped term
(→ numerical) and `> 384` → `> 383` (→ indexing).

Caveats:
- The edits are single and local. Real revisions are larger rewrites, and those get broad labels.
- The tuning-set numbers are training-set numbers.
- Detection (A vs C on the mutants) needs `akc mutants --detect --device gpu`.

Also on this day, adding a workload got simpler:
- the VOLTA layout is derived from the OpSpec (identical to the old hand-written layouts, tested);
- the launch grid is derived from the manifest;
- perf / VOLTA / tirx / tiles / numerical_dists are optional.

Unit tests: 126 (all pass on the login node, 6 GPU-only skipped).

### 8c. First GPU detection run on the mutants (2026-11-16, jobs 24048511 / 24048665 / 24050224)
Smoke job 24048511 (A100) after the revision: 126 unit tests OK (2 skipped); injected bugs A 4/11, B 5/11,
C 11/11 (CUDA) and A 1/4, C 4/4 (tirx); speed numbers unchanged. So the revision broke nothing on the GPU.

`akc mutants --detect --per-op 2` (A = fixed tests, C = fixed + change-aware tests, no VOLTA). Both jobs hit the
55-minute limit; results are in `results/eval/mutants_detect_{bases,heldout}_v1.json` (partial). What it showed:

1. **A hung kernel cost 30 minutes.** `for (off = 16; off > 0; off >>= 1)` → `off >= 0` never ends. The GPU
   runner's per-candidate timeout was 900 s, paid once by A and once by C. Fixed: timeout 180 s, and C is not
   re-run when A already caught the mutant (C includes A's tests).
2. **3 of the 8 held-out GLM drafts fail on their own** (glm_A_r2, glm_A_r3, glm_B1_r2): every mutant of them is
   "caught". Fixed: with `--detect`, each base kernel first runs the fixed tests and the floor tests; a failing
   base is classified but not used for detection.
3. **A real coverage gap.** The GLM kernels switch to a second code path when the row does not fit in shared
   memory (`use_smem = N <= SMEM_FLOATS`, 8192; tirx: `(n*4 + NWARP*4) <= 48*1024`, n ≤ 12284). The fixed suite
   stops at N = 4096 and the change-aware sizes follow the tiles, so no test ever ran that path: 4 edits there
   (`i < N` → `i <= N` twice, `[r, col]` → `[r, col + 1]`, ...) went unseen. Fixed: `extract_thresholds` finds
   comparisons of a size with constants (#define / constexpr / UPPER_CASE, evaluated by a small AST interpreter,
   no eval) and the floor set runs both sides (N = 8192 and 8193). Found in all GLM kernels that have such a
   switch, none in ours (gemm's loop variable `k` is not the size K).
4. Most other misses cannot change outputs or are invisible to tests by nature:
   - cannot change outputs: an array declared 1 larger (`red[BLOCK + 1]`), `float` → `double`
     (now marked `Mutant.bug = False`, not counted for detection);
   - equivalent in this kernel (judged by reading the code, still counted): `__shfl_xor_sync` → `__shfl_down_sync`
     or `if (lane == 0)` negated when only lane 0's value is used / all lanes hold the same value; `l = 0` → `1`
     in an online softmax whose first update multiplies l by exp(-FLT_MAX - m) = 0; shuffle start 16 → 8 in the
     second-stage reduce over NWARP = 8 values; a block size change in a kernel that derives everything from it;
   - shift-only: shuffle start 16 → 8 in the max reduction gives a smaller max; softmax is shift-invariant, so
     outputs only change if exp overflows (see §8e: this is a real float bug, and the `mixed` distribution would
     reach it, but these edits are classified as indexing and never get `mixed` inputs);
   - tests cannot see them, VOLTA's job: barrier deleted (race), `BLOCK / 2` → `BLOCK / 1` and
     `smax[4]` → `smax[2]` (out-of-bounds shared-memory access that happens to read zeros / its own writes).

Partial v1 numbers (semantic mutants, correct bases only, before the fixes above):
tuning set 92 mutants, A 49, C 63; held-out (glm_B1_r1 + glm_txl_C_r2) 34, A 17, C 22. No-op edits: 0 false
alarms in A or C on correct bases.

### 8d. Detection after the fixes (2026-11-16, jobs 24051467 / 24052255 / 24052791)
Both mutation jobs completed (31 and 24 minutes). Bases that fail by themselves are skipped (glm_A_r2, glm_A_r3,
glm_B1_r2); edits that cannot change outputs are not counted.

| kernels | edits | A | C (tests) | C + VOLTA on C's CUDA misses | no-op false alarms |
|---|---|---|---|---|---|
| tuning set | 136 | 83 | 105 | 119 | 0/23 |
| held out (5 correct bases) | 67 | 45 | 53 | 53 (tirx: VOLTA n/a) | 0/11 |

The size-regime tests caught the two `i < N` → `i <= N` edits on GLM's large-row path (glm_B1_r3 L99, glm_B1_r1
L83).

VOLTA on the 31 CUDA edits C's tests missed (`scripts/experiments/volta_on_misses.py`):
- 14 RACE / OUT_OF_BOUNDS. These are the deleted barriers, `BLOCK / 2` → `/ 1`, halved shared arrays, and
  `if (lane == 0)` negated. The last one is a same-value write race whose outputs do not change.
- 11 EQUIV over the reals. This includes the partial-max shuffle edits: VOLTA proves softmax's shift invariance
  with its symbolic exp.
- 4 UNSUPPORTED, all in glm_A_r1 (-FLT_MAX overflows VOLTA's rationals).
- 2 RACE on glm_B1_r1. That base is the known racy draft, so these are not attributable to the edit.

The 14 tirx misses (no VOLTA for tirx yet):
- 5 block-size changes that change nothing;
- 4 same-value races and 1 real race;
- 2 one-past-the-row reads inside a max pass;
- 2 changes below rtol 1e-4: `[r, col + 1]` in a 12 285-element sum, and a float16 sum accumulator (checked in
  TIRx's generated CUDA: the partial sum really is `half`).

On the LLM option: the edits C still misses are races (VOLTA), equivalent edits, sub-tolerance precision
changes, or the shift-only edits of §8e. Risky sizes from an LLM would not catch any of them, so this benchmark
cannot show its value. The real agent revisions of Tasks 4–5 are the better test.

### 8e. Correction: 8 "equivalent" misses are float bugs (2026-11-16)
Re-reading the 31 CUDA misses line by line showed that §8d's "11 EQUIV" mixes two kinds of edit.

**The 8 shift-only edits.**
- These are shuffle start 16 → 8 in the warp max reduction:
  - glm_B1_r3 L65 (const, shfl_off);
  - glm_B2_r1 L9 (const, shfl_off);
  - glm_B3_r1 L29 (const, shfl_off);
  - glm_A_r1 L65 (shfl_off; VOLTA UNSUPPORTED);
  - glm_B1_r1 L52 (shfl_off; VOLTA reports the base's race).
- Effect: the row max covers 16 of 32 lanes. This is equal over the reals, which is why VOLTA says EQUIV.
- In float32, exp overflows when the true max is >88 above the partial max.
- A CPU model of glm_B2_r1 L9 (1000 elements, element i read by thread i % 128) gave wrong results on 20 seeded
  rows per distribution:
  - `mixed`: 8 of 20 rows wrong;
  - `large`: 1 of 20;
  - every other distribution: 0;
  - a row of zeros with one 100: NaN.
- Why C missed them:
  - the classifier labels them indexing (loop-step);
  - the generated tests are normal-valued boundary shapes, plus the floor's single 2-row `large` test.

**The 9 edits that really change nothing.**
- xor → down shuffles where only lane 0 is used: glm_B2_r1 L10/L18, glm_B3_r1 L30/L49, glm_A_r1 L66/L80.
- `l = 0` → `1` in glm_A_r1. The first update multiplies l by exp(-FLT_MAX - m) = 0.
- A second-stage reduce over NWARP = 4 values starting at offset 8 (glm_B1_r1 L61).
- gemm's extra zero K-tile.

**Corrected breakdown of the 31 CUDA misses:**
- 14 caught by VOLTA;
- 8 shift-only float bugs that no validator catches;
- 9 harmless.

The tirx max-pass edits that read one element past the row (D2 L151, D3 L97) are similar. They change outputs
only when the next row starts far above this row's max: every exp underflows and the result is 0/0.

**Planned fix (not done yet):** a `spike` input distribution. Each row has one value far above the rest, at a
lane that changes from row to row. It goes in the floor set and in the numerical tests. This motivation also comes
from looking at misses, like the size-regime fix.

## D1 shape domain (2026-11-17)
The CAKE contract: the inner loop tunes one exact shape, correctness is checked across shapes and input
distributions, and the valid shape domain is declared before tuning. Ranges were decided by the team.

**What changed.**
- Manifests have a `domain` key (inclusive ranges):
  - softmax, layernorm: rows in [1, 4096], N in [1, 16384];
  - gemm: M, N, K in [1, 4096].
- The benchmark shape is still `perf.shape`: softmax and layernorm rows=4096, N=1024; gemm 1024^3.
- `Workload.domain`, `in_domain(shape)`, `clip(shape)`, `domain_text()` (`akc/task/workload.py`).
- Change-aware generator (`generate.py`):
  - a generated test or VOLTA size outside the domain is clamped to the edge; the tag says so; duplicates are dropped;
  - a size regime whose other side is outside the domain is skipped (that code path never runs inside it);
  - `extract_thresholds(..., domain)` searches only inside the domain; `plan()` passes the manifest's domain.
- LLM second opinion: risky sizes are checked against the domain instead of `DIM_CAP` (rows may now go up to
  4096 instead of 256). The prompt text changes, so earlier cached LLM replies no longer match.
- `perf.measure` returns "outside the declared shape domain" for such a shape, before any GPU work.
- `agent/pilot.py: contract_text(wl, lang)` states the range and "Speed is measured at <perf.shape>"; both pilot
  prompts (tirx and CUDA) use it. The pilot was not run.

**Without `domain`.** Nothing is limited; the regime search uses `REGIME_CAP` and the LLM check uses `DIM_CAP`,
as before. Checked: 1732 generator / threshold / prompt outputs are identical to the pre-change code with the key
removed.

**Checks.**
- Every manifest's `perf.shape`, `fixed_suite` and VOLTA default / deep shapes are inside the domain.
- With the domain, the plans of all 16 mutation-benchmark kernels (`akc/evaluation/mutants.py`) are unchanged, so
  earlier evaluation numbers stay valid. Only out-of-domain inputs change: gemm tiles >= 683 (e.g. M = 6T + 4) and
  regimes beyond the max.
- GLM size regimes are inside the range: CUDA N = 8192; tirx N = 12268, 12284, 12288 (both sides <= 16384).
- Unit tests: 144 run, all pass (6 skipped), 15 of them new.

## Task 3: VOLTA on tirx (2026-11-17)
Goal (proposal Task 3 and Task 1): run VOLTA on the compiled tirx-lite kernels the agent writes, collect verdicts,
counterexamples and diagnostics, turn them into feedback, and confirm the kernels are in VOLTA's scope.

**What changed.**
- New `akc/validation/volta/tirx.py`: trace a tirx-lite module at one size, take TIRx's generated CUDA, rewrite it
  for VOLTA, compile to PTX (`backend.ptx` mode "volta"), and read grid / block / dynamic shared bytes / tensor
  parameter order from the traced kernel.
- `layouts.py`: `make_tirx_case` (arrays from the OpSpec and the manifest's `tirx.tensors`, pointer-only params,
  each kernel's own grid and block, the larger dynamic shared size), `volta_shape`, `tirx_arrays`.
- `runner.py`: `run_analyze` (`volta analyze` of the candidate alone); "non-finite constant" now maps to UNSUPPORTED
  (it was ERROR).
- `validator.py`: a tirx path; the CUDA path is unchanged. The verdict-to-finding mapping is shared
  (`result_finding`), so `feedback/render.py` works unchanged.
- `hints.py`: hints for the new limits; for tirx, the advice names txl calls (e.g. `txl.cuda.cta_sync()`).
- `tests/validation/test_interface.py`: the old "tirx is unsupported" test now expects EQUIV for the reference.

**Steps that did not work at first, and the fixes** (each checked by running VOLTA on the generated PTX):
1. `expf` becomes `ex2.approx` + range reduction. Fix: `expf` -> `__symexpf` (the planned fix).
2. The reference itself was then UNSUPPORTED: `NEG_BIG = -3.4e38` gives "rational coefficient overflowed i128".
   - Rewriting the PTX constant to -inf was not enough: "non-finite constant -inf", because nvcc had already
     turned the max into a compare + select, which VOLTA keeps as an opaque term.
   - Rewriting the C literal to -inf works: nvcc then folds `max(-inf, x)` to `x`.
   - The rewrite is exact only for a max identity (stated in `notes/volta_scope.md`).
3. All 4 GLM kernels failed next:
   - `bar.sync %r, %r` has register operands because `txl.ptx.bar.sync(0, 128)` goes through a helper function.
     Fix: rewrite the call to `bar.sync 0` when the count equals the CTA size (exact).
   - `ex2.approx` gives "lowering failed: Unsupported instruction: Ex2". Fix: read it as exp(t / fl32(log2 e)); a
     modelling assumption.
4. GLM C (`txl.Select(a > b, a, b)` with a -FLT_MAX identity in a warp reduction) gave "non-finite constant -inf".
   Fix: a PTX peephole turns compare + select into `max.f32` / `min.f32`; exact over the reals.
5. VOLTA uses one `--dyn-shared` value for both kernels.
   - GLM D1 needs 16 bytes at N = 256, the reference 512, so a candidate overflow below 512 bytes would be hidden.
   - Fix: such a candidate is first checked alone at its own size.
   - This turned the BLOCK = 64 mutant from UNINIT into the more precise OUT_OF_BOUNDS.

**Results** (Slurm job 24060621, 3 min; details and tables in `notes/volta_scope.md`, data in
`results/eval/volta_tirx.json`):
- 23 tirx kernels x 3 sizes (N = 256, 584, 200): 69 of 69 runs decided, so coverage is 100%.
- Reference vs itself: EQUIV. All 4 GLM kernels: EQUIV.
- Injected bugs: same verdicts as their CUDA twins. The one exception is the off-by-one at N = 256, where the tirx
  version of the bug cannot trigger.
- Of the 14 tirx mutants the tests missed, VOLTA catches 8, including all 5 named races. Two earlier
  classifications were wrong:
  - the BLOCK = 64 edit of the reference is a shared out-of-bounds read, not equivalent;
  - D2 L151 / D3 L97 also overflow the staged row into the reduction scratch.
- Ablation (job 24060724, N = 256, reference + 4 GLM kernels): with all rewrites 5/5 are decided.
  - Without one rewrite: exp 0/5, ex2 1/5, huge_const 0/5, bar_count 1/5, select_max 4/5.
  - So every rewrite is needed by at least one kernel.

**Not done / open.**
- `akc eval-bugs --lang tirx` and the mutant detection with VOLTA (`--volta`) were measured before this change and
  were not re-run (they need the GPU tests). Configurations B and C now include VOLTA for tirx.
- Only softmax has tirx kernels.

**Tests.** `tests/validation/test_volta.py` has 19 new tests:
- 13 need no tools: rewrites, PTX peephole, entry parsing, case building, analyze parsing, hint choice;
- 6 run end to end and need nvcc, VOLTA and tvm_tirx.

Login node, full suite: 163 run, all pass, 6 skipped (GPU-only).

GPU-job time: 3 jobs, about 7.5 min (24060443: a first full run with older code, superseded; 24060621; 24060724).

## D3: IR change analysis + outlier inputs (2026-11-17)
The proposal: "After the agent revises a Cake IR kernel, we will inspect the modified IR operations and classify the
change". tirx-lite is our CAKE IR, so the tirx path of Task 2 now works on the traced TIRx IR instead of Python
text. CUDA (legacy) keeps the text path.

**New building blocks** (`akc/validation/changeaware/`):
- `irdiff.flatten(func, src_file) -> Flat`
  - Input: a traced `txl.Kernel.func`.
  - Output: one record per statement or header (For / If / Else / While / device_entry region), with depth, kind,
    canonical text, source line and caller line.
  - Canonical text comes from one TVMScript print of the whole function. Each statement is tagged with
    `obj_to_annotate`. The free-variable preamble is dropped.
  - Every variable is renamed in definition order (v0, v1, ...); txl calls every local `v`. Parameters and txl's
    named values (bx, thread_id) keep their names.
  - If printing fails, the text falls back to the statement's own `script()`, then `str()`, then type +
    `structural_hash`.
  - Lines come from `SourceLoc.start_line`. A helper's statement carries `CallSiteLoc(callee, caller)`; its line is
    the helper's line, and its caller line is the call site in the kernel.
- `irdiff.diff_records(a, b) -> hunks`
  - difflib runs over (depth, kind, text with variable numbers removed). One inserted local therefore does not
    renumber everything after it.
  - Variable identity is checked separately, as a bijection built from aligned definitions. `max(a, b)` →
    `max(b, a)` and a store into another buffer are caught that way.
  - For each changed pair, `get_first_structural_mismatch(map_free_vars=True)` runs per field or call argument
    and gives the exact changed node, e.g. `If.condition: LT → LE`.
  - Shifts in unrolled loops: a deleted and an added record from the same source line, kind and op are paired
    and diffed. A pure iteration shift of a non-sync statement is dropped.
- `irdiff.merge(per_shape)`
  - Groups hunks by (tag, source lines, kinds). Repeats collapse: nosync gives one hunk "×14".
  - Each merged hunk records the shapes where it appears.
- `classify_ir.classify_ir(hunks) -> [Label]`
  - Pure Python on the JSON hunks; the vocabulary is in its docstring.
  - Reduction rule: a changed shuffle, loop bound or If condition whose body does a float max / min / add is also
    numerical. In tirx-lite the Python loop of a tree reduction is unrolled at trace time, so its bound shows up
    as If conditions.
  - `classify.proposal_category()` maps indexing → tiling, so reports can use the proposal's 4 categories. The 5
    internal labels are unchanged.
- `ir_worker`
  - Runs in a child process: `python -m akc.validation.changeaware.ir_worker`, with a 60 s limit per revision (plus
    a SIGALRM per job).
  - `backend.tirx.safety_scan` runs first (also in the parent, so an unsafe or unparsable source never starts a
    process). Then `load_module` and the manifest factory trace both versions at every shape.
  - Returns hunks and labels as JSON.
  - `prefetch()` analyses many revisions in one child; the mutation benchmark uses it.
- `trace_shapes`: all inside the domain, rows = 2:
  - `perf.shape`;
  - N = 1, T−1, T+1;
  - v and v+1 of each size regime (either version);
  - N = 2T+1, then further tiles;
  - at most 8.

**`plan()`.** For tirx with a parent, it runs the IR path. `ChangePlan.analysis` holds:
- `mode`: ir | ast-fallback | cuda-text;
- the traced shapes and skipped shapes;
- hunk counts (ir, per shape, ast);
- `fallback_reason` and `ir_blind`.

Rules:
- tracing fails, times out or is refused → AST labels, with the reason;
- IR equal at every shape but an AST change → AST labels, `ir_blind = True`;
- AST cosmetic but IR changed → IR.

Other details:
- identical sources are not traced;
- `classifier="ast"` forces the old path;
- the LLM option still sees source-line hunks.

The validator puts `analysis` in its artifacts, and `ctx.extra["classifier"]` selects the path.

**Checks on the injected tirx bugs** (CPU, 5 shapes each: N = 1024, 1, 127, 129, 257):

| bug | IR hunks | labels |
|---|---|---|
| offbyone | 1: `If.condition LT → LE`, L51 → L54 | indexing guard + numerical reduction-bound |
| tail | 2: `For.end` at L41, L49, not at N = 1024 | tiling trip-count + numerical reduction-bound |
| nosync | 1 del hunk at L34, ×14, called from L46 and L54 | sync barrier |
| nomax | 2: `Sub → TensorLoad` inside exp, L53 and L60 | numerical |

- Re-trace and a renamed local give no hunks.
- `exp(v - mx)` → `exp(mx - v)` → numerical operand.
- x → y in the max loop → memory buffer.
- The float16 mutants of the GLM kernels do not trace → AST fallback, reason recorded.
- Time: one revision 7.7 s (mostly the tvm import); in a batch 1.1–1.4 s per revision. The AST path takes 0.01 s.

**IR vs AST on the mutation benchmark** (tirx rows only; `akc mutants --classifier ir|ast --only ...`, CPU):
- The AST verdicts were re-run first: identical to `mutants_rules_after_*.json`.
- IR results:
  - tirx reference (tuning set): 21/21 right, 0/3 no-op labelled; 2 AST fallbacks, 0 `ir_blind`;
  - 4 GLM tirx kernels (held out): 70/70 right, 0/12 no-op labelled; 8 AST fallbacks, 0 `ir_blind`.
- Agreement matrix: 91 hit/hit, 15 clean/clean, nothing else.
- Same category set 70/106; same first category 90/106.
- IR mostly adds categories: +numerical 29, +memory 12, +indexing 8. The shuffle offset `16 → 8` in a warp max is
  sync + numerical (AST: indexing).
- Generated tests: 1409 vs 1027 (+37%).
- **Not fully held out.** The first IR run scored 69/70 held out. Its miss (glm_txl_D3_r1 L114, `red_max[w]` →
  `[w + 1]` in an unrolled loop) led to the same-line shift pairing above.
- Results: `results/eval/mutants_rules_ir_{bases,heldout}.json`. `--compare-with` prints the matrix.

**Outlier inputs** (`akc/task/inputs.py`, `make_tensor(shape, "outlier@P", gen)`; TestCase and its JSON
unchanged):
- each row (last dimension) is N(0, 1) times a row scale of 1 or 1e3, alternating between adjacent rows;
- each row has one outlier at column (r mod P) + P·k, k seeded-random (any column if N < P). Plain `outlier` means
  P = N;
- outlier magnitude: ladder[(r // 2) % 4] × row scale, ladder = +1e2, +1e3, −1e3, +1e4;
- why these magnitudes: at least 20× the largest N(0, 1) value in a row of ≤ 16384 elements; three decades, plus a
  negative rung; at most 1e7, so squares and K = 4096 dot products stay far below float32's limit.

Where it is used:
- in `DISTS` and `DEFAULT_DISTS`, and in every manifest's `numerical_dists`, as the 2nd entry: never first, so the
  floor tests never use it;
- the generator writes `outlier@{tiles[0]}` with rows ≥ P (softmax / LayerNorm: 128 × 1000; GEMM: P = 16);
- `sanitizer_trial.py` uses `outlier@128` instead of the removed `spike` (2 lines; not re-run).

The idea came from looking at benchmark misses (§8e), so any benchmark gain is tuning-set evidence only.

**Do correct kernels pass the outlier inputs?**
- CPU, float32:
  - checked with the harness `compare()` and the op's tolerance, for `outlier@P` and plain `outlier`, 3 seeds;
  - shapes: rows 128–256 with N = 129, 1000, 4096 and 16384; GEMM 128×128×256 and 64×64×4096;
  - PyTorch float32 passes for all 3 ops, and so does a block-128 strided + tree-reduction model of softmax and
    LayerNorm.
- NumSim (tirx), `outlier@128` at N = 129 and 1000, plain `outlier` at N = 256: the tirx reference,
  glm_txl_D1_r2 and glm_txl_D3_r1 all pass.
- NumSim allows at most 4 rows, so on the CPU the generated `outlier@128` test (128 rows) is reported "skipped"
  for tirx. It only runs fully on the GPU.

**Tests.**
- New: `tests/validation/test_changeaware_ir.py` (13; 8 need `tvm_tirx`) and `tests/task/test_outlier.py` (7).
- Login node, full suite: 196 run, all pass, 8 skipped. A run earlier the same day showed 1 failure in
  `test_volta.py`; that is the parallel VOLTA work and was gone in the final run.

**GPU check of the outlier inputs** (job 24061002, A100, 2 min 14 s; `build/tmp/d3_outlier/run.sh`, `akc check`).
- Tests:
  - softmax / LayerNorm: `outlier@128` at 128×1000, 256×4099 and 128×129, plus plain `outlier` at 64×256;
  - GEMM: `outlier@16` at 128×128×256 and 64×80×1000;
  - tirx: `outlier@128` at 128×1000 and 128×129.
- Every correct kernel passes:
  - softmax / LayerNorm / GEMM `_ref` and `_opt`;
  - glm_A_r1, glm_B1_r3, glm_B2_r1, glm_B3_r1 (CUDA);
  - softmax_txl_ref and glm_txl_{C_r2, D1_r2, D2_r1, D3_r1}.
- For information, the injected CUDA bugs on these inputs alone:
  - softmax nosync, offbyone and nomax: 0/4 pass; tail: 1/4;
  - LayerNorm offbyone: 0/4; tail: 1/4; nosync and onepass pass all 4;
  - GEMM ktail: 1/2; nosync and guard pass.
- Results: `results/harness/check_{softmax,layernorm,gemm}_20261117-03*.json`.
- Not done: detection on the mutation benchmark with the new inputs (A vs C on the GPU).

## Task 1: tirx LayerNorm + GEMM (2026-11-17)
Goal (proposal Task 1): the agent writes tirx-lite only. So every workload needs:
- a tirx-lite reference with baseline correctness and speed;
- VOLTA able to analyze the kernels.

Before this, only softmax had tirx kernels.

**What changed.**
- New kernels, plain SIMT for sm_80 (no tensor cores, TMA or cp.async):
  - `workloads/layernorm/tirx/layernorm_txl_ref.py`: thread 0 of a 32-thread CTA does the whole row, two-pass,
    like `layernorm_ref.cu`. tirx-lite blocks have at least 32 threads.
  - `workloads/layernorm/tirx/layernorm_txl_opt.py`: block-parallel two-pass, 128 threads, shared-memory tree
    reductions (like `layernorm_opt.cu`). eps = 1e-5 is a constant: no scalar parameters, so VOLTA can read it.
  - `workloads/gemm/tirx/gemm_txl_ref.py`: one thread per output.
  - `workloads/gemm/tirx/gemm_txl_opt.py`: 16 x 16 shared-memory tiling with zero-padded tiles.
  - Both GEMMs use 256 threads read as 16 x 16 and a 2-D grid (ceil(N/16), ceil(M/16)), the same tile mapping as
    the CUDA tile2d launch. The tirx launch is inside the kernel (`LaunchConfig`), so the shared GPU runner needed
    no change.
- Bug ports, single edits of the opt kernel with the same meaning as the CUDA bugs:
  - LayerNorm: nosync (barrier after reading red[0] removed), tail (n // 128 tiles), onepass (E[x^2] - mean^2),
    offbyone.
    - The offbyone port covers col = 0..n: loop count (n + 128) // 128 and guard col <= n. This is the tirx form of
      CUDA's `i <= N`, so it reads the next row's first element at every N.
    - The softmax tirx off-by-one only changes the guard, so it shows only when N % 128 != 0.
  - GEMM: ktail (k // 16 tiles), guard (row <= m, col <= n), nosync (barrier after the inner product removed).
- Manifests:
  - layernorm and gemm: `tirx` blocks (tensors x, w, b, y / A, B, C), `references.tirx`, a tirx signature, and the 6 /
    5 variants with `role`, `bug` and `expect_volta`. Each `expect_volta` is identical to its CUDA twin's.
  - softmax: `expect_volta` on its 4 tirx bugs.
    - nosync RACE; nomax EQUIV;
    - tail and offbyone: EQUIV if N % 128 == 0, else NOT_EQUIV.
- `akc/validation/runners/numsim.py`: size caps also for M (<= 64) and K (<= 512); before, a GEMM fixed test
  (128^3) would have been simulated. The skip message names the cap.
- `scripts/experiments/volta_tirx.py`:
  - `--workload`;
  - an `opt` group;
  - expectations from the variant's own `expect_volta` (the CUDA twin's as fallback);
  - full-shape `--shapes "M=..,N=..,K=..;..."`;
  - default shapes = manifest default + deep + change-aware tiling / sync shapes.
  - The Task 3 softmax run used `--shapes 256,584,200`.
- New `scripts/experiments/tirx_task1.py`:
  - runs GPU tests, NumSim, VOLTA and perf for every tirx variant of layernorm and gemm, with NumSim and VOLTA on
    the CPU in parallel with the GPU parts;
  - merges into `results/eval/tirx_task1.json`.
- New `scripts/slurm/tirx_task1.sh`: the job.
- Tests:
  - new `tests/validation/test_tirx_kernels.py`, 13 tests:
    - manifests and variants;
    - bugs are 1-2 line edits;
    - VOLTA layouts for the new tensors (CPU);
    - tracing, launch and parameter order (tvm_tirx);
    - VOLTA end to end (nvcc + VOLTA);
    - 2 GPU tests.
  - `test_volta.py`: `test_workload_without_tirx_reference_is_unsupported` used gemm as the workload without a
    tirx reference; it now strips the tirx parts of gemm's manifest.
- Not changed: `runners/gpu.py`, `volta/tirx.py`, `volta/layouts.py`, `perf/measure.py`, `akc bench`.
  - TIRx orders kernel parameters by name (b, w, x, y); `build_tirx_ptx` already maps them through the traced
    function's parameter names.

**Steps that did not work at first.**
1. The first LayerNorm tirx reference failed the offset test (x ~ 1e4 + N(0, 1)) on the A100 (job 24061076).
   - The plain serial float32 sum drifts by about 1e-2 on such rows.
   - Fix: mean = x0 + sum(x - x0) / n. It is equal over the reals; VOLTA still proves EQUIV.
   - A CPU float32 emulation checked it first: max err 0.0168 -> 6.4e-4.
   - The legacy CUDA `layernorm_ref` fails the same test (job 24061108); it is unchanged.
2. A NumSim debug run on the login node, gemm 12 x 12 x 40, took 2.5 min (6.5 CPU-min). So NumSim ran in the GPU
   job instead. There, after the first engine build (about 115 s), each kernel took 7-11 s.

**Results** (jobs 24061076 and 24061108; tables in `notes/gpu_results.md` and `notes/volta_scope.md`):
- GPU tests:
  - ref and opt pass all tests (LayerNorm 15, GEMM 17);
  - every bug except LayerNorm nosync fails at least one test (that one is also missed on CUDA).
- NumSim: ref / opt pass; every bug is caught.
  - The gemm guard bug shows as runtime_error: NumSim stops at the out-of-range store with an "unmodeled analysis
    contract" message.
- VOLTA: LayerNorm 18/18, GEMM 25/25 as expected; coverage 100%.
- Softmax tirx bugs at the new shapes: 24/25. softmax_txl_bug_tail at N = 100 is UNSUPPORTED:
  - no tile runs, so the -inf max identity stays in the expression;
  - VOLTA is undecided there, not wrong.
- Speed, x PyTorch: layernorm ref 0.13, opt 1.07-1.12; gemm ref 0.12-0.13, opt 0.28-0.29.

**Not done / open.**
- `akc check` and `akc volta` (`cli.py`, Task 2's file) still handle CUDA variants only.
  - For tirx use `scripts/experiments/tirx_task1.py` or `scripts/experiments/volta_tirx.py --workload W`.
- `akc eval-bugs --lang tirx` has not been re-run. It would now include the 7 new bugs with the tirx opt as parent.
- Mutation benchmark: the new kernels are not added. Suggested: `layernorm_txl_opt` and `gemm_txl_opt` as HELDOUT
  bases, since the classifier rules were never tuned on them.

GPU-job time: 2 jobs, about 10 min 15 s (24061076: 4 min 56 s; 24061108: 5 min 19 s).

## D2: compiler-checks gate (2026-11-17)
The proposal: "Candidates that pass CAKE's existing compiler checks will be checked by VOLTA." tirx-lite is our CAKE
substitute, so its own checks are the static half of the gate. The CAKE paper also feeds conventional sanitizer
output back to the agent, so compute-sanitizer is the dynamic half.

**Configurations** (`akc/validation/pipeline.py`):
- A0 = fixed tests only (the ablation; the same validator and context as the old A);
- A = compiler-checks gate + fixed tests;
- B = A + VOLTA (default shape);
- C = B + change-aware tests (VOLTA at the change-aware shapes);
- order in C: compiler_checks, fixed_tests, changeaware, volta.

**New building blocks** (`akc/validation/compiler_checks/`):
- `tirx_static.gate_shapes(workload, src)`:
  - row ops: rows = 4 with N ∈ {T+1, 1000} (T = the candidate's tile), plus rows = 2, N = v+1 for each
    `extract_thresholds` value v;
  - gemm: 33×33×40 and 64×64×64;
  - clipped to and filtered by the manifest domain.
- `tirx_static.run_static(workload, module, shapes, out_dir, numsim=None)`:
  - one child process (`python -m akc.validation.compiler_checks.tirx_static`); the parent safety-scans first;
    SIGALRM of 60 s per shape;
  - per shape: trace (`check_ir=True`, so `LowLevelIRContractError` is raised while tracing),
    `inspect_low_level_ir` again (catches `@txl.kernel(check_ir=False)`), `check_well_formed`, `verify_memory`,
    `compile_sm80`;
  - the generated CUDA is read from the compile result (`ex.mod.imports[0].inspect_source("cuda")`, identical to
    `Kernel.source()`), so each shape compiles once; entry name, tensor order, grid, block and dynamic shared bytes
    come from `volta.tirx` (`kernel_tensor_params`, new, factored out of `build_tirx_ptx`; `launch_info`);
  - contract violations are located at Python lines: trace again with the check switched to record-only, then match
    the violating statements with `irdiff.flatten` (the violation itself has no span);
  - optional NumSim stage (CPU-only node): racecheck + synccheck at `numsim_shape` (rows = 2, N = T+1; gemm
    12×12×20) on .npy arrays from the parent; each finding keeps the Python lines of both accesses and the call site.
- `sanitizer.py`:
  - `build_cubin`: `nvcc -cubin -arch=sm_80 -lineinfo` of the generated CUDA;
  - `write_inputs`: OpSpec inputs and sentinel-filled outputs as .npy, in kernel-parameter order;
  - `run_tools`: memcheck, racecheck, initcheck, synccheck, one compute-sanitizer process each, in parallel; each
    runs every gate shape; 60 s limit per tool;
  - `parse_sanitizer`: the trial's block parser plus kind, locations, PCs and attribution by file name;
  - `sass_line_map`: `nvdisasm -gi` of the cubin, SASS offset → line of the generated CUDA in the kernel body.
- `launcher.py`: the child every tool wraps; run by path, imports only numpy and cuda.bindings (importing anything
  under `akc.validation` pulls in torch). Exact-size `cuMemAlloc` per tensor; outputs left uninitialized under
  initcheck; outputs of the memcheck run are saved for a value check in the parent.
- `validator.py`: `CompilerChecksValidator` ("compiler_checks"):
  - static error → `compile_error` (error) with the Python line; the dynamic stage does not run;
  - sanitizer report of ours → race / out_of_bounds / uninitialized / deadlock (error), racecheck INFO hazards →
    info; a kernel fault no report explains → `runtime_error`; a tool crash / timeout → `harness_error` (warning);
  - NumSim error → error, review → warning, incomplete → info `unsupported`;
  - artifacts: stages (seconds + status), shapes, tools, per-tool result, logs, value check;
  - legacy CUDA: one info finding "not part of A/B/C".

**Pipeline** (`run_pipeline(cand, config, ctx, cache=None, skip=())`):
- a `compile_error` always stops; any other failure stops unless `ctx.extra["run_all"]`;
- with the gate in the configuration, the test validators get `ctx.extra["numsim_racecheck"] = False`, and the
  NumSim test runner skips Racecheck (the gate ran it on the CPU);
- `cache`: a validator with the same inputs in two configurations of one candidate runs once (key: validator,
  candidate and parent source, device, VOLTA shapes, the NumSim switch for tirx on the CPU, relevant extra keys);
- `PipelineResult`: `per_validator` (caught / seconds / decided / kinds / cached), `stopped_after`, `skipped`,
  `cached`, `validation_seconds`.

**Other changes:**
- `evaluation/bugset.py`: `evaluate(cand, ctx, configs)` runs A0 / A / B / C through `run_pipeline` with `run_all`
  and one cache per bug; per configuration caught + seconds + per-validator numbers; `--configs`; saves after every
  bug.
- `evaluation/mutants.py`:
  - `_detect` uses `evaluate` (A0 / A / C, plus B with `--volta`; without `--volta` VOLTA is skipped);
  - `_base_ok` also requires the gate to pass;
  - new set `fresh` (`FRESH`: layernorm_txl_ref / opt, gemm_txl_ref / opt), `all` = bases + heldout + fresh;
    `--lang all|cuda|tirx`; each row records its set;
  - generation gaps fixed: `const` no longer edits the exponent of a float literal (`EPS = 1e-5` → `1e-6` was
    labelled as a structural edit); the fresh set also gets `init` on `txl.float32(<value>)` initializers and `smem`
    on allocations sized by a named constant (`alloc([BLOCK])` → `alloc([BLOCK // 2])`). The extended operators are
    off for bases / heldout: their mutants are byte-identical to before (64 fingerprints checked, per_op 1, 2, 3, 20).
- `feedback/render.py`: a `location:` line for every finding that has one; gate hints for race / out_of_bounds /
  uninitialized / deadlock / runtime_error; for tirx without a Python line it says "in the CUDA that TIRx generated
  from your module (no Python line is known)" and quotes the line (and the other access of a race).
- `cli.py`: `akc validate --config A0|A|B|C [--run-all]`; `akc check --lang tirx`; `akc volta --lang tirx`
  (`evaluation/volta_matrix.py`, through VoltaValidator's tirx path).
- `runners/numsim.py`, `runners/__init__.py`: the Racecheck switch.

**Steps that did not work at first** (GPU smoke job 24061664 and full job 24061708):
1. racecheck reported the softmax and LayerNorm nosync races, but the gate dropped them: the reports name the
   inlined TIRx helper (`tvm_builtin_ptx_st_shared_f32(unsigned int, float)+0x260`), not the kernel, and the trial's
   parser keeps only reports that name the kernel. Fix: a report is ours if it names the kernel or a line of one of
   our generated .cu files.
2. The reported lines were the helpers' `asm` lines. Fix: `nvdisasm -gi` maps the reported offset to the call site
   in the kernel body. Checked on the saved log: all 10 offsets land on the STS / LDS of the reported accesses, and
   the race becomes "line 116 (write) and line 112 (read)", two steps of the tree reduction.
3. memcheck with the trial's `--destroy-on-device-error kernel` reported the out-of-bounds read of two GLM mutants
   and then hung until the 180 s timeout. Fix: `--destroy-on-device-error context` (later shapes of that run are
   skipped) and a 60 s tool limit. Job 24061708 was cancelled after 33 of 34 kernels to re-run with the fix.

**GPU results** (job 24061793, `results/eval/gate_tirx.json`; tables also in `notes/gpu_results.md`):

| kernels | memcheck | racecheck | initcheck | synccheck | gate (any) |
|---|---|---|---|---|---|
| 5 tirx refs / opts (false alarms) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 4 GLM tirx kernels (false alarms) | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| 11 injected tirx bugs | 3/11 | 3/11 | 0/11 | 0/11 | 6/11 |
| 8 tirx mutants with VOLTA RACE / OUT_OF_BOUNDS | 2/8 | 3/8 | 2/8 | 0/8 | 4/8 |
| 6 tirx mutants with VOLTA EQUIV | 1/6 | 0/6 | 0/6 | 0/6 | 1/6 |

- The memcheck run's outputs match the float64 reference on all 9 correct kernels at every gate shape: the launcher
  (argument order, grid, block, dynamic shared bytes) is right.
- Same detection as the CUDA trial on the injected bugs (6/11; the 5 value bugs need tests or VOLTA), and again
  `layernorm_txl_bug_nosync`, which every GPU test misses, is caught by racecheck.
- The 4 missed race mutants are `if (lane == 0)` negations: same-value writes, harmless (as in the CUDA trial).
- One catch VOLTA cannot make: glm C idx1 L114 reads one element past `x` on the N > 12284 path (gate shape
  N = 12285 from the size-regime rule); VOLTA checks CTA 0 at N ≤ 584 and said EQUIV.
- Cost per revision (median of 34): 10.9 s (max 12.7 s): static child 7.8 s (tvm import 1.7 s, two sm_80 compiles
  2.8 s), nvcc 0.7 s, the four tools in parallel 2.1 s (sequential 4.5–4.7 s on 3 kernels, same verdicts).
- CPU-only path (smoke job 24061664): gemm nosync → race at Python lines 50 and 42, gemm guard → out_of_bounds at
  line 56, gemm opt clean; 12 s per kernel after a 44 s first NumSim engine build.

**A0 / A / B / C on the 11 injected tirx bugs** (`akc eval-bugs --lang tirx --device gpu`, job 24061793,
`results/eval/bugset_tirx_gpu_20261117-045938.json`):
- detected: A0 3/11, A 6/11, B 6/11, C 11/11;
- validation seconds per bug (mean): A0 8.6, A 18.6, B 21.4, C 41.8;
- per validator in C: gate 6/11 (10.0 s), fixed tests 3/11 (8.6 s), change-aware tests 11/11 (20.9 s), VOLTA 4/11
  (2.2 s);
- the report cache works as intended: in C the gate and fixed-test reports are reused from A (`cached`).
- **Found: VOLTA in C runs at B's default shape for tirx.** `changeaware.plan()` returns VOLTA shapes only for CUDA
  (`volta_shapes(...) if lang == "cuda" else []`, unchanged since the restructure), so `suggested_volta_shapes` is
  empty for every tirx candidate. C's detection is unaffected here (the change-aware tests catch all 11), but VOLTA's
  column in C is understated. Not fixed (Task 2 code, outside D2).
- A CPU run of `eval-bugs --lang tirx --workloads gemm --configs A0,A` before the GPU job: A0 0/3 (the gemm fixed
  suite exceeds NumSim's caps, so "n/a"), A 2/3 (NumSim race on nosync, out_of_bounds on guard). Not kept.

**Fresh mutation set, CPU** (`akc mutants --set fresh --classifier ir`, `results/eval/mutants_rules_ir_fresh.json`):
- 85 mutants of the 4 fresh kernels in 70 s; 73/73 semantic edits get a right category, 0 wrong, 0 without a
  category; 0/12 no-op edits labelled;
- 79 on the IR path, 6 AST fallbacks: 5 `float32 → float16` edits and BLOCK 32 → 16 in layernorm_txl_ref, which
  TIRx refuses to trace (the gate reports such edits as compile errors);
- detection on the GPU (`--set fresh --detect`) was not run.

**Tests.**
- New `tests/validation/test_compiler_checks.py` (28): parser on saved logs (fixtures in `tests/data/sanitizer/`,
  plus every trial log in `logs/sanitizer_trial/` when present: same verdicts as the trial on all of them); a real
  TIRx racecheck log mapped to kernel lines 116 / 112 through a rebuilt cubin; finding kinds; NumSim and static
  mappings; feedback text; gate shapes; pipeline order, stop rules, run_all, cache, skip (fake validators); A0 = the
  old A; static checks on the 5 refs / opts and 11 bugs pass and an IR-contract violation fails at its line (with
  and without `check_ir`); 3 GPU tests.
- `test_mutants.py` +3 (fresh set, extended operators, float exponents), `test_bugset.py` +1 (evaluate),
  `test_interface.py`: C's report list now starts with compiler_checks.
- Login node: 228 run, all pass, 11 skipped. A100: 46 run in job 24061708 (all pass, 1 skipped), the 3 GPU gate tests
  again in job 24061793 after the memcheck change.

**Verified vs assumed.**
- Verified on the A100: the gate's verdicts and costs above; the value check of every correct kernel; no hang with
  `--destroy-on-device-error context` on the two kernels that hung before.
- Verified on the CPU: static checks on all 16 tirx variants, contract violations, NumSim stage on softmax (login
  node) and gemm (job 24061664); `akc check --lang tirx` and `akc volta --lang tirx` (one small run each, outputs
  deleted); fresh-set generation and classification; existing mutant sets unchanged.
- Assumed: that running the 4 tools at the same time on one GPU does not change what they report (3 kernels gave the
  same verdicts both ways; the CUDA trial ran 6 at a time).
- Not done: `akc mutants --set fresh --detect` on the GPU; the CUDA sanitizer path inside the gate (legacy CUDA gets
  an info finding only).

GPU-job time for D2: 3 jobs, 31 min 37 s (24061664: 1 min 32 s; 24061708: 13 min 54 s, cancelled; 24061793:
16 min 11 s).
