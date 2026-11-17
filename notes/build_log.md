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
