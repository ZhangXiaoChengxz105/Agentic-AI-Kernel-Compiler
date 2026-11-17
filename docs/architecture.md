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
                 │ pipeline(config A | B | C)                                              │
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
| `agent/` | the agent itself | `pilot.py` (working prototype). Task 4 adds `prompts/`, `context.py`, `loop.py`, `trajectory.py` |
| `backend/` | executes actions | `ptx.py` (CUDA → PTX, VOLTA/GPU modes), `launch.py` (driver-API launch), `bench.py`, `tirx.py` (safety scan, load, sm_80 compile) |
| `validation/` | the critics | `interface.py`, `pipeline.py`, `fixed_tests/`, `volta/`, `changeaware/`, `runners/` |
| `perf/` | the objective (speed) | `measure.py` (GPU timing in a child process), `report.py` (`PerfReport`, roofline `analyze`), `hardware.py` (GPU peaks) |
| `feedback/` | observations | `render.py`: findings → concise text with hints, examples, related change lines; `render_perf` (speed), `render_observation` (both) |
| `evaluation/` | experiments | `bugset.py` (A/B/C on injected bugs), `volta_matrix.py`, `gpu_tirx.py` |

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
| `fixed_tests/` (baseline, A) | manifest `fixed_suite` | runs power-of-two N(0,1) tests ×3 | one finding per failing test |
| `volta/` (mechanism 1, Task 3) | `ctx.volta_shapes` or manifest default | `volta compare` vs the workload reference: equivalence over ℝ + races/deadlocks, CTA 0 | per-shape verdicts in `artifacts.runs`; `hint` in evidence; tirx → info "unsupported" |
| `changeaware/` (mechanism 2, Task 2) | `parent` vs candidate | diff (reformat / rename = no change) → classify (tiling / indexing / memory / sync / numerical) → targeted tests + always-on floor tests + VOLTA shapes → run tests; optional LLM second opinion (`ctx.extra["llm_classifier"]`: extra labels + risky sizes, union only) | `artifacts.categories`, `labels`, `suggested_volta_shapes`, `llm`; findings carry `related_change` |

`runners/` executes TestCases for both test validators: `gpu.py` (CUDA via driver API, tirx via TIRx compile;
sentinel + canaries + repeats; each candidate in a child process) and `numsim.py` (tirx on the CPU, capped).

### Configurations (`validation/pipeline.py`)
```
A: fixed_tests
B: fixed_tests + volta(default shape)
C: fixed_tests + changeaware + volta(change-aware shapes)
```
A compile error stops the pipeline. `run_pipeline(cand, "C", ctx)` returns a `PipelineResult`.

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
The agent loop (Task 4) keeps the fastest candidate that passed validation and returns it at the end.
CLI: `akc bench --workload W [--lang tirx] [--feedback]`, `akc validate ... --perf` (validation, then speed).

## Workloads (`workloads/<op>/`)
Data, not code: `manifest.json` + `cuda/` (+ `tirx/`) kernels and `bugs/`. Manifest keys:
`op, description, shape_keys, launch {kind: row|tile2d, block, tile}, tiles, fixed_suite, numerical_dists,
volta {default, deep, max_N}, tirx {factory_args, tensors}, perf {shape}, references {cuda, tirx}, signature, variants[]`
(`variants`: name, lang, file, role, block, bug {category, change}, expect_volta).

Adding a workload = a manifest (required keys: op, description, shape_keys, launch, fixed_suite, variants) + kernels
+ an `OpSpec` in `task/reference.py` (make_inputs, golden, out_shape, kernel_args, atol/rtol). Optional add-ons, each
reporting "not set up" when absent:
- VOLTA: `references.cuda` + `volta` in the manifest. The memory layout is derived from the OpSpec
  (`volta/layouts.py: derive_layout`).
- speed: `perf.shape` + OpSpec flops/bytes/torch_impl.
- tirx-lite: `tirx`.
- `tiles` / `numerical_dists`: have defaults.
- a custom grid: OpSpec `launch`; the default is derived from the manifest's launch kind.

Limit: `changeaware/generate.py` has size patterns for (rows, N) and (M, N, K) workloads only.

Kernel conventions: `extern "C" __global__`, write `exp` as `EXP(x)` (`workloads/common/common.cuh`;
`-DVOLTA` turns it into VOLTA's `__symexpf` hook), control flow must not depend on data values.

## Command line (`akc …`, after `source env.sh`)
`validate` (pipeline on one kernel; `--perf` adds speed) · `hottest` (change-aware report for a revision) · `check` / `bench` (GPU) ·
`volta` · `eval-bugs` · `mutants` (classifier / detection benchmark) · `tirx-gpu` · `pilot` · `lumen`. Slurm wrappers: `scripts/slurm/`.

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
- Detection of those edits (A100): fixed tests 61% → fixed + change-aware 77% (tuning set), 67% → 79% (held out).
  Of the 31 remaining CUDA misses:
  - 14 are races or out-of-bounds shared access, and VOLTA catches all of them;
  - 9 change nothing;
  - 8 make the max reduction skip half the warp. These are equal over the reals, but give NaN in float32 on a
    row with one large value. No test input has such a row yet; a `spike` distribution is the planned fix.
- Size regimes: `extract_thresholds` only sees comparisons with compile-time constants; a threshold computed at
  run time (e.g. from `cudaDeviceProp`) is not found.
