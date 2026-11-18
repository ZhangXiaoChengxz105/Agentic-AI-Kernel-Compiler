# akc architecture

akc (Agentic Kernel Compiler) is an agent system: an LLM writes GPU kernels, **validators** check every
revision, and their findings become the **observation** that drives the next revision. The project's
contribution is the validation layer: a fixed-test baseline plus two complementary mechanisms, VOLTA formal
verification and change-aware testing (proposal sections 1 and 2).

```
                 ┌────────────────────────────── agent ──────────────────────────────┐
  task ────────► │ context (what the model sees) → llm (GLM-5.3 via Lumen) → kernel  │
  (workload,     └───────────────▲───────────────────────────────────────┬──────────┘
   reference,                    │ feedback.render(reports)               │ Candidate
   constraints)                  │                                        ▼
                 ┌───────────────┴──────────────── validation ─────────────────────────────┐
                 │ pipeline(config A0 | A | B | C)                                         │
                 │   compiler_checks (gate: static checks + sanitizer / NumSim)             │
                 │   fixed_tests ─┐                                                         │
                 │   changeaware ─┼─► runners (GPU harness / TIRx NumSim) ─┐                │
                 │   volta ───────┴────────────────────────────────────────┴► ValidationReport │
                 └──────────────────────────────────────────────────────────────────────────┘
                                 │ passed? ──► perf.measure (GPU, same in A/B/C) ──► PerfReport ──► feedback.render_perf
                                 │ trajectory (every iteration) ──► evaluation (A/B/C metrics)
```

## Packages (`akc/`)
| package | role in the agent | contents |
|---|---|---|
| `task/` | the problem | `workload.py` (manifest loader), `reference.py` (PyTorch golden + launch/args per op), `inputs.py` (value distributions), `testcase.py` |
| `llm/` | the model | `lumen.py`: OpenAI-compatible client, budget guard, retries, JSONL call log |
| `agent/` | the agent itself | `loop.py` (the repair loop, Task 4), `prompts.py`, `context.py`, `trajectory.py`; `pilot.py` (earlier prototype) |
| `backend/` | executes actions | `ptx.py` (CUDA → PTX, VOLTA/GPU modes), `launch.py` (driver-API launch), `bench.py`, `tirx.py` (safety scan, load, sm_80 compile) |
| `validation/` | the critics | `interface.py`, `pipeline.py`, `compiler_checks/`, `fixed_tests/`, `volta/`, `changeaware/`, `runners/` |
| `perf/` | the objective (speed) | `measure.py` (GPU timing in a child process), `report.py` (`PerfReport`, roofline `analyze`), `hardware.py` (GPU peaks) |
| `feedback/` | observations | `render.py`: findings → concise text with hints, examples, related change lines; `render_perf` (speed), `render_observation` (both) |
| `evaluation/` | experiments | `bugset.py` (A0/A/B/C on injected bugs), `mutants.py` (mutation benchmark; `--detect`), `volta_pool.py` (VOLTA in parallel CPU workers for the evaluation; merged reports = the serial validator's), `volta_matrix.py`, `gpu_tirx.py` |

Agent "tools" (functions the model may call itself, e.g. "compile this draft") are optional and would live in
`agent/tools.py`; they are different from `backend/`, which is infrastructure used by validators and the loop.
Validators always run inside the loop: the agent under test does not choose whether it gets checked, and
configurations A/B/C must apply the same checks every iteration.

## The validator contract (`validation/interface.py`)
Every validator implements `validate(cand: Candidate, ctx: ValidationContext) -> ValidationReport`.

| type | fields |
|---|---|
| `Candidate` | `workload`, `lang` ("cuda" / "tirx"), `source`, `entry`, `block`, `parent`, `label`, `path` |
| `ValidationContext` | `config`, `test_cases`, `volta_shapes`, `device` ("auto" / "gpu" / "cpu"), `time_budget_s`, `extra` |
| `Finding` | `kind`, `severity` (error / warning / info), `validator`, `message`, `evidence`, `location` |
| `ValidationReport` | `validator`, `passed` (no error findings), `decided`, `findings`, `seconds`, `artifacts` |

- `Finding.kind` ∈ compile_error, runtime_error, race, deadlock, not_equivalent, mismatch, nonfinite,
  out_of_bounds, unwritten, uninitialized, footprint, div_by_zero, nondeterministic, unsupported, timeout,
  skipped, harness_error. VOLTA's 12 statuses and the harness's 8 failure reasons all map onto these.
- `decided` feeds **verification coverage** (did the validator reach a verdict?), `seconds` feeds
  **validation overhead** (proposal's evaluation metrics).
- **Parent policy**: `Candidate.parent` is the last candidate that *passed* validation, not just the previous
  revision; the first candidate is compared with the workload reference. (GLM pilot: a bug introduced in round 2
  survived a round-3 edit that only touched an initializer.)

### Validators
| validator | input it uses | what it does | output specifics |
|---|---|---|---|
| `compiler_checks/` (the gate, A/B/C; D2) | the tirx-lite candidate | static checks in a child process (`tirx_static.py`: safety scan, trace with the IR contract check, `inspect_low_level_ir`, `check_well_formed`, `verify_memory`, compile for sm_80) at small gate shapes; then on a GPU compute-sanitizer memcheck / racecheck / initcheck / synccheck on cubins built from the generated CUDA (`sanitizer.py`, torch-free `launcher.py`), on a CPU-only node NumSim racecheck / synccheck at one small shape | static error → `compile_error` (Python line when known); sanitizer report naming our kernel → race / out_of_bounds / uninitialized / deadlock with the generated-CUDA line; tool crash → `harness_error` warning; NumSim error / review / incomplete → error / warning / info. `artifacts.stages` (seconds + status), `sanitizer` (per tool), `value_check`, logs |
| `fixed_tests/` (baseline, A0 / A) | manifest `fixed_suite` | runs power-of-two N(0,1) tests ×3 | one finding per failing test |
| `volta/` (mechanism 1, Task 3) | `ctx.volta_shapes` or manifest default | `volta compare` vs the workload reference: equivalence over ℝ + races/deadlocks, CTA 0 | per-shape verdicts in `artifacts.runs`; `hint` in evidence; tirx: traced per shape, generated CUDA rewritten for VOLTA (`volta/tirx.py`), compared with `references.tirx` |
| `changeaware/` (mechanism 2, Task 2) | `parent` vs candidate | diff (tirx-lite: the traced IR, `irdiff.py` in a child process via `ir_worker.py`; CUDA: normalized text; reformat / rename = no change) → classify (tiling / indexing / memory / sync / numerical; tirx by the IR op that changed, `classify_ir.py`) → targeted tests + always-on floor tests + VOLTA shapes → run tests; optional LLM second opinion (`ctx.extra["llm_classifier"]`: extra labels + risky sizes, union only) | `artifacts.categories`, `labels`, `suggested_volta_shapes`, `llm`, `analysis` (ir / ast-fallback / cuda-text, traced shapes, fallback reason, `ir_blind`); findings carry `related_change` |

**Change analysis on the IR (D3).** For tirx-lite, both versions are safety-scanned, traced at up to 8 shapes
inside the declared domain, and compared statement by statement:
- variables are renamed in definition order;
- `get_first_structural_mismatch` names the changed IR node;
- hunks are merged across shapes and unrolled repeats.

If tracing fails, the Python-AST path is used and the reason is recorded. Reports may map indexing → tiling
(`proposal_category()`) to show the proposal's 4 categories.

**Value inputs.** `task/inputs.py` has a generic `outlier@P` family:
- one outlier per row at column ≡ r (mod P);
- adjacent rows scaled 1 / 1e3.

The numerical tests use it with P = the kernel's tile.

`runners/` executes TestCases for both test validators: `gpu.py` (CUDA via driver API, tirx via TIRx compile;
sentinel + canaries + repeats; each candidate in a child process) and `numsim.py` (tirx on the CPU, capped).

### Configurations (`validation/pipeline.py`)
```
A0: fixed_tests                                                   (ablation: no gate; = the old A)
A : compiler_checks + fixed_tests                                 (CAKE's existing checks + tests)
B : compiler_checks + fixed_tests + volta(default shape)
C : compiler_checks + fixed_tests + changeaware + volta(change-aware shapes)
```
`run_pipeline(cand, "C", ctx, cache=None, skip=())` returns a `PipelineResult` (`per_validator`: caught / seconds /
decided / kinds / cached; `stopped_after`, `skipped`, `cached`, `validation_seconds`).
- A `compile_error` (a static error of the gate, or a build failure) always stops the pipeline.
- Any other failure also stops it by default (the agent loop gets the first failure fast). With
  `ctx.extra["run_all"]` the later validators still run; the evaluation sets it to count per validator.
- With the gate in the configuration, the NumSim test runner skips Racecheck (`ctx.extra["numsim_racecheck"]`):
  on a CPU-only node the gate already ran NumSim racecheck / synccheck.
- `cache` (one dict per candidate): a validator with identical inputs in two configurations runs once; the
  evaluation (`evaluation/bugset.evaluate`, `mutants._detect`) runs A0 / A / B / C this way.

### The compiler-checks gate (`validation/compiler_checks/`, D2)
The proposal: "Candidates that pass CAKE's existing compiler checks will be checked by VOLTA." tirx-lite is our CAKE
substitute, so its own checks are the static half of the gate. The CAKE paper also feeds conventional sanitizer
output back to the agent, so compute-sanitizer is the dynamic half.
- **Shapes** (`tirx_static.gate_shapes`, inside the manifest domain): row ops rows=4 × N ∈ {T+1, 1000}, plus
  rows=2 × N=v+1 for each size-regime value v of the candidate; GEMM 33×33×40 and 64×64×64.
- **Static stage** (one child process; tvm import ~2–5 s, then per shape: trace ~0.1 s, the 4 IR checks ~3 ms,
  compile 1.5–2 s): `txl.kernel(check_ir=True)` raises `LowLevelIRContractError` while tracing;
  `inspect_low_level_ir` runs again explicitly (so `check_ir=False` does not bypass it); violations are located at
  Python lines by re-tracing with the check in record-only mode and matching statements (`irdiff.flatten`).
- **Sanitizer stage (GPU)**: the generated CUDA (`Kernel.source()`, read from the sm_80 compile) → `nvcc -cubin
  -lineinfo`; parent writes inputs as .npy; `launcher.py` (numpy + cuda.bindings only, run by path so torch is never
  imported) allocates every tensor exactly, launches with the traced grid / block / dynamic shared bytes. The 4
  tools run in parallel, one process each; each runs every gate shape. Reports are filtered by kernel name and
  attributed to a shape by the generated file name. Locations are lines of the generated CUDA: TIRx emits no Python
  line info, so the feedback says so and quotes the line.
- **NumSim stage (CPU-only node)**: racecheck + synccheck at one small shape (`numsim_shape`: rows=2 × N=T+1;
  GEMM 12×12×20); findings carry the Python lines of both accesses.

## Speed: the objective, not a validator (`perf/`)
Validation is a pass/fail gate and is what configurations A/B/C vary. Speed has no pass/fail and must be
measured **identically in A/B/C**, so it lives outside `validation/` and runs only for candidates that passed.
VOLTA cannot help here: it proves equality over the reals and has no notion of time.

`measure(cand, shape=None) -> PerfReport` (GPU; child process like the test runner):
1. correctness at the benchmark shape (manifest `perf.shape`), because a fast wrong kernel is meaningless and the
   validated sizes may differ;
2. kernel time with `backend/bench.py`: warmup, CUDA events, L2 flushed before every launch, median of 50;
3. PyTorch's own implementation under the same conditions (`speedup_vs_torch`);
4. a device copy of the op's minimum bytes (the practical floor for memory-bound ops).

| `PerfReport` field | meaning |
|---|---|
| `measured`, `reason` | false + why: no GPU, compile error, incorrect at the benchmark shape, crash, timeout |
| `time_ms` (`p10_ms`, `p90_ms`), `torch_ms`, `speedup_vs_torch` | median time; > 1 = faster than PyTorch |
| `bytes`, `flops`, `intensity`, `bound` | minimum DRAM traffic, arithmetic, FLOP/byte; memory / compute / latency (< 10 µs) |
| `achieved_gbps`, `pct_peak_bw`, `achieved_tflops`, `pct_peak_flops` | effective rates vs the datasheet peaks (A100-40GB: 1555 GB/s, FP32 19.5 TFLOP/s, ridge 12.5 FLOP/byte) |
| `roofline_ms`, `floor_ms`, `headroom` | hardware lower bound; practical floor (copy time, or roofline when compute-bound); time / floor |

`feedback.render_perf(report, best)` turns it into text (example from the A100 run numbers):
```
Speed at rows=4096,N=1024 on A100-SXM4-40GB (median of 50 launches, cold L2): 0.0363 ms = 1.02x PyTorch's speed ...
Memory-bound op (0.62 FLOP/byte, below the GPU's ridge of 12.5): effective bandwidth 924 GB/s = 59% of the 1555 GB/s peak.
A plain device copy of the same 33.6 MB takes ... ms: that is the practical floor, ...x faster than this kernel.
Best correct version so far: r1 at 0.0343 ms, 1.08x PyTorch; this one is 1.19x its time.
```
The agent loop (Task 4) keeps the fastest candidate that passed validation and returns it at the end (see below).
CLI: `akc bench --workload W [--lang tirx] [--feedback]`, `akc validate ... --perf` (validation, then speed).

## The agent loop (`agent/`, Task 4)
`run_agent(workload, config, llm=client, max_iters=10, optimize=False, seed=None, ...) -> AgentResult`
(`akc agent --workload W --config C [--seed F] [--optimize] [--dry-run]`). The agent writes tirx-lite.
```
[seed kernel -> iteration 0, no model call]          optional: e.g. an injected bug, to measure repair
for it = 1 .. max_iters (10):
    messages = context.build_messages(...)           system + task, then: progress table, previous draft + its
                                                     observation, best correct version so far, instruction
    reply = llm.chat(messages)                       coin cap per run (worst case checked before each call)
    cand  = Candidate(code, parent = last accepted)  first candidate: compared with the workload reference
    res   = run_pipeline(cand, config)               the same checks every iteration
    perf  = perf.measure(cand) if res.passed
    observation = render_observation(res.reports, perf, best perf)
    trajectory.record(step)                          every iteration, as it happens
    stop if res.passed (or, with optimize, continue for speed until the cap)
```
- `prompts.py`: the task prompt is built from the manifest (description, `signature.tirx`, `tirx.factory_args` /
  `tensors`, shape contract, OpSpec tolerance), plus the tirx-lite README, two examples and a pitfalls list. The
  reference kernels are never shown.
- `context.py`: the context is rebuilt every turn, not grown as a chat (GLM has no prompt-cache discount). Only
  the latest draft and the best correct version are resent.
- Stop reasons: `passed`, `max_iters`, `budget`, `llm_error`, `validation_error` (a validator crashed), `dry_run`.
- Best candidate: the fastest measured one that passed; without a GPU, the first that passed.
- `trajectory.py` writes `results/agent/<timestamp>_<workload>_<config>/`: `run.json`, `trajectory.jsonl` (one line
  per iteration: verdict, failing validators, error kinds, per-validator seconds, perf, tokens, coins, timings),
  per iteration `it<k>_prompt.json`, `it<k>_reply.md`, `it<k>.py`, `it<k>_validation.json`,
  `it<k>_observation.txt`, then `best.py` and `summary.json` (passed, stop reason, first-pass iteration, coins,
  validation / model seconds).

## Workloads (`workloads/<op>/`)
Data, not code: `manifest.json` + `cuda/` (+ `tirx/`) kernels and `bugs/`. Manifest keys:
`op, description, shape_keys, launch {kind: row|tile2d, block, tile}, tiles, fixed_suite, numerical_dists,
volta {default, deep, max_N}, tirx {factory_args, tensors}, domain, perf {shape}, references {cuda, tirx}, signature, variants[]`
(`variants`: name, lang, file, role, block, bug {category, change}, expect_volta).

Adding a workload = a manifest (required keys: op, description, shape_keys, launch, fixed_suite, variants) + kernels
+ an `OpSpec` in `task/reference.py` (make_inputs, golden, out_shape, kernel_args, atol/rtol). Optional add-ons, each
reporting "not set up" when absent:
- VOLTA: `references.cuda` + `volta` in the manifest. The memory layout is derived from the OpSpec
  (`volta/layouts.py: derive_layout`).
- speed: `perf.shape` + OpSpec flops/bytes/torch_impl.
- shape domain: `domain` `{key: [lo, hi]}`, the CAKE contract (valid range declared before tuning; speed at
  `perf.shape`). `Workload.in_domain` / `clip`: change-aware tests, VOLTA shapes, size regimes and LLM sizes stay
  inside it, `perf.measure` rejects a shape outside it, the agent prompt states it. Absent = no limit.
- tirx-lite: `tirx`.
- `tiles` / `numerical_dists`: have defaults.
- a custom grid: OpSpec `launch`; the default is derived from the manifest's launch kind.

Limit: `changeaware/generate.py` has size patterns for (rows, N) and (M, N, K) workloads only.

Kernel conventions: `extern "C" __global__`, write `exp` as `EXP(x)` (`workloads/common/common.cuh`;
`-DVOLTA` turns it into VOLTA's `__symexpf` hook), control flow must not depend on data values.

## Command line (`akc …`, after `source env.sh`)
`validate` (pipeline on one kernel; `--perf` adds speed) · `hottest` (change-aware report for a revision) · `check` / `bench` (GPU) ·
`volta` · `eval-bugs` · `mutants` (classifier / detection benchmark) · `tirx-gpu` · `agent` · `pilot` · `lumen`. Slurm wrappers: `scripts/slurm/`.

## Where each task works
| task | modules |
|---|---|
| 1 environment & baseline | `setup_env.sh`, `backend/`, `task/`, `validation/runners/`, `validation/fixed_tests/` |
| 2 change-aware testing | `validation/changeaware/` |
| 3 VOLTA integration | `validation/volta/`, `feedback/` (source-line mapping, counterexamples, tirx path) |
| 4 agent loop | `agent/` (uses `validation.run_pipeline`, `perf.measure`, `feedback.render_observation`, `llm`) |
| 5 evaluation | `evaluation/`, `perf/` (final kernel performance), `workloads/*/…/bugs/` |

## Known limits (details in `notes/`)
- VOLTA: real-number semantics; CTA 0 only; exact sizes only; needs `EXP()` and PTX without `-lineinfo`;
  constants above ~1.7e38 overflow its i128 rationals; LayerNorm VCs grow ~N² (keep N ≤ 200).
- GPU tests are weak at races (VOLTA is the main race check); NumSim is slow per new shape.
- The injected bugs and the classifier were written by the same person. The classifier is also measured on
  generated edits (`akc mutants`, including held-out GLM kernels: 189/189).
- Detection on tirx-lite (A100, 2026-11-17; README §4): 164 generated edits, A0 73% → A 84% → B 90% → C 93%;
  11 injected bugs 3 → 6 → 6 → 11; no false alarms on 27 no-op edits. 11 of C's 12 misses cannot change outputs.
  (Legacy CUDA edits: fixed tests 61% → fixed + change-aware 77% on the tuning set.)
- VOLTA decides about 86% of the tirx-lite edits it runs on; a candidate whose block 0 covers another region than the
  reference's (another GEMM tile size) is undecided.
- A TIRx-compiled function refuses misaligned tensors, so the change-aware generator makes misaligned-pointer tests
  for legacy CUDA only.
- Size regimes: `extract_thresholds` only sees comparisons with compile-time constants; a threshold computed at
  run time (e.g. from `cudaDeviceProp`) is not found.
