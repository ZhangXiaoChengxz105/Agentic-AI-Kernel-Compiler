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
