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
| `feedback/` | observations | `render.py`: findings → concise text with hints, examples, related change lines |
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
| `changeaware/` (mechanism 2, Task 2) | `parent` vs candidate | diff → classify (tiling / indexing / memory / sync / numerical) → targeted tests + VOLTA shapes → run tests | `artifacts.categories`, `labels`, `suggested_volta_shapes`; findings carry `related_change` |

`runners/` executes TestCases for both test validators: `gpu.py` (CUDA via driver API, tirx via TIRx compile;
sentinel + canaries + repeats; each candidate in a child process) and `numsim.py` (tirx on the CPU, capped).

### Configurations (`validation/pipeline.py`)
```
A: fixed_tests
B: fixed_tests + volta(default shape)
C: fixed_tests + changeaware + volta(change-aware shapes)
```
A compile error stops the pipeline. `run_pipeline(cand, "C", ctx)` returns a `PipelineResult`.

## Workloads (`workloads/<op>/`)
Data, not code: `manifest.json` + `cuda/` (+ `tirx/`) kernels and `bugs/`. Manifest keys:
`op, description, shape_keys, launch {kind: row|tile2d, block, tile}, tiles, fixed_suite, numerical_dists,
volta {default, deep, max_N}, tirx {factory_args, tensors}, references {cuda, tirx}, signature, variants[]`
(`variants`: name, lang, file, role, block, bug {category, change}, expect_volta).

Adding a workload = a manifest + kernels, an `OpSpec` in `task/reference.py` (golden + kernel args + launch),
and a memory layout in `validation/volta/layouts.py`.

Kernel conventions: `extern "C" __global__`, write `exp` as `EXP(x)` (`workloads/common/common.cuh`;
`-DVOLTA` turns it into VOLTA's `__symexpf` hook), control flow must not depend on data values.

## Command line (`akc …`, after `source env.sh`)
`validate` (pipeline on one kernel) · `hottest` (change-aware report for a revision) · `check` / `bench` (GPU) ·
`volta` · `eval-bugs` · `tirx-gpu` · `pilot` · `lumen`. Slurm wrappers: `scripts/slurm/`.

## Where each task works
| task | modules |
|---|---|
| 1 environment & baseline | `setup_env.sh`, `backend/`, `task/`, `validation/runners/`, `validation/fixed_tests/` |
| 2 change-aware testing | `validation/changeaware/` |
| 3 VOLTA integration | `validation/volta/`, `feedback/` (source-line mapping, counterexamples, tirx path) |
| 4 agent loop | `agent/` (uses `validation.run_pipeline` + `feedback.render` + `llm`) |
| 5 evaluation | `evaluation/`, `workloads/*/…/bugs/` |

## Known limits (details in `notes/`)
- VOLTA: real-number semantics; CTA 0 only; exact sizes only; needs `EXP()` and PTX without `-lineinfo`;
  constants above ~1.7e38 overflow its i128 rationals; LayerNorm VCs grow ~N² (keep N ≤ 200).
- GPU tests are weak at races (VOLTA is the main race check); NumSim is slow per new shape.
- The change classifier and the injected bugs were written by the same person: re-measure on agent-written
  revisions and third-party bugs.
