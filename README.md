# Agentic-AI-Kernel-Compiler (akc)

CS 521 project (Hongyi Chen, Minghao Fang, Xiaocheng Zhang). An LLM agent writes GPU kernels. Every revision is
checked by **VOLTA** formal verification and by **change-aware tests** generated from what the revision changed.
The findings, plus the kernel's measured speed, become the feedback for the next revision. We compare these
validation configurations:

- **A0**: fixed tests only (ablation without the gate);
- **A**: compiler-checks gate + fixed tests (CAKE's existing checks: tirx-lite static checks + compute-sanitizer);
- **B**: A + VOLTA;
- **C**: B + change-aware tests.

The proposal's "candidates that pass CAKE's existing compiler checks will be checked by VOLTA" is the gate (§2 D2).

Status (2026-11-17): **the whole chain runs on tirx-lite**, the agent's language (proposal: the agent writes CAKE IR;
tirx-lite stands in for CAKE, see §2). Tasks 1–3 are done for three kernels (softmax, LayerNorm, GEMM): baseline,
change-aware testing on the IR, VOLTA on the compiled tirx-lite kernels, and the compiler-checks gate that runs
first. Task 5's detection / coverage / overhead experiment has been run on tirx-lite (§4). Task 4 (the repair loop)
is implemented (`akc/agent/loop.py`, `akc agent`, §2 Task 4); runs with the real model and repair-success numbers are
still open.

More detail: [`docs/architecture.md`](docs/architecture.md) (module contracts), [`docs/setup.md`](docs/setup.md)
(setup troubleshooting), `notes/` (dated experiment logs; `notes/gpu_results.md` has the A100 runs).

---

## 1. Environment

### Requirements
| item | version / value | notes |
|---|---|---|
| cluster | NCSA Delta | everything was tested there |
| GPU | NVIDIA A100-SXM4-40GB, sm_80, driver 595.71.05 | only on compute nodes, through Slurm |
| CUDA | nvcc 13.2 (`module load cudatoolkit/26.5_13.2`, loaded by default) | CUDA 13 still supports sm_80 |
| Python | 3.13 (`module load python/3.13.5-gcc13.3.1`) | env built in `envs/akc` |
| Rust | stable (builds VOLTA) **and** 1.95.0 (TIRx NumSim) | both installed in `tools/` |
| libz3 | 5.4.1 (from pip `z3-solver`) | VOLTA links against it |
| VOLTA | `third_party/volta` @ c091559 | built from source |
| key Python packages | torch 2.14.1+cu132, apache-tvm-tirx 0.1.0rc2, tirx-harness 0.1.3rc0, tirx-kernels 0.1.3rc1, cuda-bindings 13.5.0, numpy 2.5.5 | exact lock: `setup/requirements-akc.lock` |
| disk | ~10 GB (the Python env alone is ~6 GB) | |
| network | internet access during setup | |

### Setup
```bash
git clone git@github.com:ZhangXiaoChengxz105/Agentic-AI-Kernel-Compiler.git && cd Agentic-AI-Kernel-Compiler
bash setup_env.sh --tests   # builds everything inside this folder (~9 min from scratch, verified in a fresh clone)
source env.sh               # in EVERY new shell: paths, caches, libz3, activates envs/akc
bash setup_env.sh --check   # re-verify an existing setup later
```
The script is idempotent: it skips finished steps and logs to `logs/setup_<timestamp>.log`.
LLM calls (`akc agent`, `akc pilot`, `akc lumen`) use your own Lumen key: put `export LUMEN_API_KEY=...` in
`.secrets/lumen.env` (git-ignored; `env.sh` loads it).

### Notes
- **Everything is installed inside the cloned folder.** That covers Rust, libz3, VOLTA, the Python env and all
  caches; nothing goes to `$HOME`.
- **Run `source env.sh` in every new shell.** Without it:
  - VOLTA fails with `libz3.so: cannot open shared object file`;
  - `akc` is not on the PATH.

---

## 2. What we have done

### CAKE substitute
**CAKE has no public code** (checked 2026-11-15: paper, GitHub, the authors' and organisations' repositories).
In its place we use **tirx-lite** from MLC's TIRx-harness. It plays the same role, a "CAKE-style" kernel IR plus
compiler, and it works on sm_80.

Following the proposal, **the agent writes tirx-lite only**: change analysis works on the tirx-lite IR, the gate is
tirx-lite's own checks plus compute-sanitizer on the CUDA it generates, and VOLTA checks that compiled code. The
hand-written CUDA kernels and their results are **legacy**: they were our first baseline, they stay in the repository,
but they get no new features and are not part of configurations A0/A/B/C.

### Task 1: environment and baseline
- **One-command environment.** `setup_env.sh` builds VOLTA, Rust, libz3, TIRx and torch inside the cloned folder.
  It was tested in a fresh clone (see §4).
- **Three kernels.** softmax, LayerNorm and GEMM, all fp32.
  - **tirx-lite (the agent's language), all three:** `workloads/<op>/tirx/` holds a simple reference
    (`<op>_txl_ref.py`, VOLTA's comparison kernel) and the injected bugs (`bugs/`). LayerNorm and GEMM also have an
    optimized correct version (`<op>_txl_opt.py`: block-parallel two-pass LayerNorm; 16×16 shared-memory GEMM), a
    false-alarm control. They are plain SIMT code for sm_80: no tensor cores, TMA or cp.async.
  - The bugs are ports of the CUDA ones with the same meaning: softmax 4, LayerNorm 4 (tail, offbyone, onepass,
    nosync), GEMM 3 (ktail, guard, nosync).
  - The hand-written CUDA kernels in `workloads/<op>/cuda/` (a reference, an optimized version, bugs) are legacy.
- **Reference results.** PyTorch computes the correct answer in float64 (`akc/task/`). Inputs come from 8 value
  distributions: normal, uniform, large, negative large, offset, tiny, mixed and constant. Test sizes include sizes
  that are not multiples of the tile size.
- **Correctness tests.** The GPU harness (`akc/validation/runners/`) runs a kernel and compares its output with the
  reference.
  - The output buffer is pre-filled with a sentinel value, so elements the kernel never wrote are detected.
  - Canary values around the buffer detect writes past its end.
  - Each test is repeated, so results that change between runs (a sign of a race) are detected.
  - Each kernel runs in a child process, so a crashing kernel cannot take the caller down.
  - The tolerance depends on the op and on how sensitive its result is to rounding.

  The fixed test suite (every configuration has it) is in `akc/validation/fixed_tests/`. tirx-lite kernels can also run on the CPU
  through NumSim.
- **Speed measurement.** `akc/perf/` times a kernel with CUDA events: warmup, L2 cache flushed before every launch,
  median of 50 runs. It compares the time with three things:
  - PyTorch;
  - the A100's peak bandwidth and FLOP/s;
  - a device copy of the same bytes, which is the practical floor.
- **VOLTA check.** VOLTA can analyze all three tirx-lite kernels (Task 3 below): every run on the references,
  optimized versions and bugs is decided, and the verdicts match the manifest's expectations
  (`results/eval/tirx_task1.json`, `notes/volta_scope.md`). The legacy CUDA kernels: 32/32 verdicts as expected.
- **Slurm scripts and LLM client.** `scripts/slurm/` holds the GPU jobs. `akc/llm/lumen.py` is the Lumen client:
  budget guard, dry-run, retries, and a log of every call.

### Task 2: change-aware test generation
When the agent submits a new version of a kernel, `akc/validation/changeaware/` compares it with the last version
that passed validation and tests what changed:
1. **Diff.** What changed between the two versions.
   - **tirx-lite: on the IR** (`irdiff.py`, run by `ir_worker.py`), as the proposal asks ("inspect the modified IR
     operations"). Both versions are safety-scanned, traced in a child process (60 s limit) and compared statement
     by statement at up to 8 shapes inside the declared domain, all with rows = 2:
     - the benchmark shape;
     - N = 1, T−1, T+1 and 2T+1 (T = the kernel's tile);
     - both sides of each size-regime switch.

     Variables are renamed in definition order (txl calls every local `v`), so a renamed local gives no change.
     For each changed statement, TVM's `get_first_structural_mismatch` names the exact IR node, e.g.
     `If.condition: LT → LE` or `For.end: 2 → 1`. Hunks are merged across shapes and repeats: the 14 barriers
     that an unrolled loop drops become one hunk "×14". A hunk that appears only at some shapes says so; the tail
     bug, for example, does not show at N = 1024.
   - **Fallbacks** (recorded in the plan's `analysis` field):
     - tracing fails or times out → the Python-AST diff, with the reason;
     - the IR is equal at every traced shape but the AST shows a real change → the AST labels are kept and
       `ir_blind` is set;
     - the AST says "cosmetic" but the IR differs → the IR wins.
   - **CUDA** (legacy) is still compared as normalized text. Pure reformatting and consistent renaming of local
     variables count as no change.
2. **Classify.** Each change is labelled tiling, indexing, memory, sync or numerical, with the line as evidence.
   - tirx-lite (`classify_ir.py`): the label follows the IR op that changed.
     - sync: barriers, shuffles, mbarrier, fences, async copies, atomics;
     - tiling: `For.end`, the grid / block, shared-memory shapes, `smem_required`;
     - indexing: `If.condition`, loop start / step, integer index expressions;
     - memory: ld / st form, scope, dtype, `ptr_byte_offset`, a different buffer;
     - numerical: math calls, float arithmetic, float constants, casts.
   - A changed shuffle or loop bound whose body does a float max / min / add (a reduction) is also numerical, so it
     gets the value-range tests.
   - CUDA (`classify.py`): the text rules. Loop and `if` headers are split off first, so a changed loop bound, loop
     step or guard is told apart from a changed statement.
   - Reports can use the proposal's 4 categories: `proposal_category()` maps indexing → tiling.
3. **Generate tests** (`generate.py`). Each label gets its own tests, and tile sizes are read from the kernel itself:
   - tiling / indexing → sizes at tile boundaries and sizes that are not multiples of the tile (1, T±1, 2T±1, 1000;
     for GEMM also K % T ≠ 0);
   - memory → odd sizes (legacy CUDA also: pointers misaligned by one element; a TIRx-compiled function refuses
     misaligned tensors before the kernel runs, so tirx-lite gets no such test);
   - sync → large sizes, repeated runs, many vs. few thread blocks;
   - numerical → large, negative, offset, tiny, mixed and constant values, plus the **outlier family** below.

   It also picks VOLTA sizes for which thread block 0 crosses a tile boundary, plus a larger "deep" size that runs
   unrolled loops.

   **Outlier inputs** (`outlier@P` in `akc/task/inputs.py`; generic, no operator knowledge):
   - every row (last dimension) is N(0, 1) times a row scale that alternates 1 / 1e3 between adjacent rows, so a
     read from the neighbouring row changes the result far beyond tolerance;
   - each row has exactly one outlier, at column (r mod P) + P·k with k random. P is the kernel's own tile and the
     test has at least P rows, so every position modulo the tile holds an outlier in some row;
   - outlier magnitude, relative to the row scale: +1e2, +1e3, −1e3, +1e4 by row. Even the smallest is more than
     20× every other value in its row, and squares and long dot products stay far below float32's limit;
   - correct kernels pass:
     - on the A100 (job 24061002, 2 min): the CUDA reference and optimized kernels of all 3 ops, the 4 correct GLM
       CUDA kernels, the tirx reference and the 4 GLM tirx kernels;
     - on the CPU: PyTorch float32 and a block-128 reduction model, and NumSim on 3 tirx kernels;
     - NumSim allows at most 4 rows, so on the CPU the generated 128-row test is skipped for tirx;
   - the idea came from looking at mutation-benchmark misses, so gains on that benchmark are tuning-set evidence.

   A small **floor set** runs on every revision whatever the classifier says: sizes 1, T−1, T+1 and 1000, plus one
   extreme value distribution. A misclassified edit still meets a non-divisible size. The outlier family is never
   the floor's distribution: it is never first in a manifest's `numerical_dists`.

   The floor set also covers **size regimes**. Some kernels take a different code path for large rows, e.g. GLM's
   `use_smem = (N <= SMEM_FLOATS)` with 8192. The fixed suite stops at N = 4096, so the second path is never run.
   `extract_thresholds` finds comparisons between a size and constants, and the floor set runs both sides
   (N = 8192 and 8193). The constants come from `#define`, `constexpr` and UPPER_CASE names. They are evaluated by a
   small arithmetic interpreter; kernel text is never passed to `eval`.
4. **Run and report** (`validator.py`) runs the tests. Each failure carries the changed line it relates to, so the
   agent sees which edit caused it.

**Optional LLM second opinion** (`llm_classify.py`; off by default, because every call spends Lumen coins):
- It asks a model (default GLM-5.3, low reasoning effort) for categories per changed line, plus up to 4 "risky
  sizes" where this particular edit should misbehave.
- The answer is merged with the rules' labels: it can only add tests, never remove them.
- A bad or failed reply falls back to the rules.
- Replies are cached, so re-running an experiment is free.
- Turn it on with `--llm-classify` on `akc validate`, `akc hottest` or `akc mutants`.
- Not yet tried with real calls.

Example: `softmax_txl_bug_tail`. The bug changes two loop counts from `iters` (= ceil(N / 128)) to `n // BLOCK`,
so the max and sum loops skip the last N % 128 elements.
- **IR change:** the two loops' `For.end`, changed only at traced sizes that are not multiples of 128.
- **Classification:** tiling + numerical (the loops are a max and a sum reduction).
- **Tests:** sizes around the tile (1, 127, 129, 255, 257, 1000, …) plus value-range tests. On the A100 they fail
  with wrong and non-finite outputs (job 24062207).
- **VOLTA:** sizes N = 256, 584, 200 and 100. VOLTA proves the kernel wrong at N = 584 and 200. At the default
  N = 256, a multiple of 128, it says "equivalent", which is why configuration B misses this bug. N = 100 is
  UNSUPPORTED.

The unit tests are in `tests/validation/test_changeaware.py` (text / AST path, generator),
`tests/validation/test_changeaware_ir.py` (IR path; skipped without `tvm_tirx`) and `tests/task/test_outlier.py`.
How well the classifier does is measured in §4 ("Change classifier").

### D2: compiler-checks gate (CAKE's existing checks), 2026-11-17
The proposal: "Candidates that pass CAKE's existing compiler checks will be checked by VOLTA." tirx-lite is our CAKE
substitute, so the gate (`akc/validation/compiler_checks/`) is tirx-lite's own checks plus the conventional
sanitizer feedback that the CAKE paper also uses. It runs first in configurations A, B and C.
- **Static checks** (one child process per candidate):
  - safety scan of the module;
  - trace with the IR contract check (`txl.kernel(check_ir=True)` raises `LowLevelIRContractError`);
  - `inspect_low_level_ir` again, explicitly, so `check_ir=False` cannot bypass it;
  - `check_well_formed` and `verify_memory` (tvm_tirx);
  - compile for sm_80.

  Any error is a `compile_error` and stops the pipeline. Contract violations are located at the Python line.
- **Shapes** (small, inside the manifest domain): row ops rows = 4 × N ∈ {T+1, 1000}, plus rows = 2 × N = v+1 for
  each size regime v of the candidate; GEMM 33×33×40 and 64×64×64.
- **GPU: compute-sanitizer** memcheck, racecheck, initcheck and synccheck, in parallel, on a cubin built from the
  CUDA that TIRx generates (`Kernel.source()`). The launcher imports no torch and gives every tensor an exact-size
  allocation.
  - A report becomes race / out_of_bounds / uninitialized / deadlock, with the line of the generated CUDA.
  - TIRx puts every load, store and barrier in an inlined helper, so the reports name the helper. The gate maps
    the PC back to the call site in the kernel through the cubin's inline line table (`nvdisasm -gi`), and the
    feedback says that the line is in generated CUDA and quotes it.
  - A tool that crashes or times out is a `harness_error` warning.
- **CPU-only node: TIRx NumSim** racecheck and synccheck at one small shape (rows = 2 × N = T+1; GEMM 12×12×20).
  NumSim errors are errors (with the Python lines of both accesses), "review" is a warning, "incomplete" is info.
  The NumSim test runner then skips its own Racecheck.
- **Stop rules.** A compile / static error always stops the pipeline. Any other failure stops it too, unless
  `ctx.extra["run_all"]` is set; the evaluation sets it so every validator's catches can be counted.
- **Legacy CUDA** candidates get one info finding: they are not part of A/B/C.
- **Results on the A100**: see §4 "Compiler-checks gate".

### Task 3: VOLTA on the compiled tirx-lite kernels
`VoltaValidator` (`akc/validation/volta/`) traces the candidate and the workload's tirx-lite reference at each VOLTA
size, rewrites the CUDA that TIRx generates so VOLTA can read it (`volta/tirx.py`), runs `volta compare` (and
`volta analyze` for out-of-bounds accesses past the candidate's own shared memory), and turns each verdict into a
finding with a hint for the agent (`volta/hints.py`). Details: `notes/volta_scope.md`.
- **All three kernels are analyzable:** softmax 69 of 69 runs decided (23 kernels × 3 sizes), LayerNorm 18 of 18,
  GEMM 25 of 25 (`results/eval/volta_tirx*.json`). Verdicts match the CUDA counterparts.
- **Sizes:** B uses the workload's default size; C uses the change-aware sizes (`changeaware.plan().volta`): the
  default, a "deep" size that runs unrolled loops, and sizes that put a tile boundary inside thread block 0.
- **Rewrites:** three are exact (`expf` → `__symexpf`, a counted barrier → `bar.sync 0`, compare-select → `max.f32`);
  two are modelling choices: `ex2.approx(t)` is read as exp(t / log2 e), and a constant of magnitude ≥ 2^127
  (-3.4e38 as the identity of a max) becomes -inf.
- **Limits:** real-number arithmetic; thread block 0 only, so a candidate whose block 0 covers a different part of
  the output than the reference's (e.g. GEMM with another tile size) is undecided; exact sizes only; no
  data-dependent control flow, partial-CTA barriers, async copies, or transcendental functions other than exp.

### Task 4: agent feedback and repair loop (2026-11-17)
`akc/agent/loop.py` (`akc agent`) connects the model, the validation configurations and the feedback
(details: `docs/architecture.md`, "The agent loop"):
- **The loop.** Each iteration: model → tirx-lite module → `run_pipeline(config)` → `perf.measure` (only if it
  passed) → `render_observation` → next prompt. The same checks run every iteration.
- **Stopping.** At most 10 iterations. The loop stops at the first candidate that passes the configuration's
  checks; with `--optimize` it keeps improving speed until the cap and returns the fastest correct version.
- **Parent policy.** The parent is the last accepted candidate.
- **Seed kernel.** `--seed F`: a given kernel (e.g. an injected bug) is iteration 0, so repair can be measured.
- **Budget.** A coin cap per run (`--max-coins`, worst case checked before each call). `--dry-run` sends nothing.
- **Prompt.** Built from the manifest, so it works for all three workloads (`prompts.py`). The context is rebuilt
  each turn (task + previous draft + its observation + best correct version) instead of growing a chat.
- **Logging.** Every iteration is written to `results/agent/<timestamp>_<workload>_<config>/` (`trajectory.py`).
- **Tests.** Unit tests: `tests/agent/test_loop.py` (fake model and validators, 10 tests).

**Integration check on the A100** (job 24078880, 3 min). It used the real pipeline and speed measurement with a
scripted model (no coins): the seed is an injected bug, and the model's one reply is the workload's tirx reference.

| workload / config | seed (iteration 0) | caught by | it1 | speed of it1 |
|---|---|---|---|---|
| softmax / C | `softmax_txl_bug_tail` | change-aware tests (fixed tests passed): NaN/Inf at N = 1, mismatch at N = 1000, with the related IR change lines | passed C | 0.0318 ms, 1.03× PyTorch |
| LayerNorm / B | `layernorm_txl_bug_nosync` | gate (racecheck), with the generated-CUDA lines | passed B | 0.296 ms, 0.13× |
| GEMM / A | `gemm_txl_bug_nosync` | gate (racecheck) | passed A | 1.09 ms, 0.12× |

All three runs stopped with `passed` after one model turn. Validation took 9–50 s per iteration and speed
measurement ~6 s. No run with the real model yet: it needs Lumen coins (a GLM tirx run cost 0.07–0.17 coins in
the pilot).

### Open items
- **Repair success (Task 4/5) is not measured yet.** The detection, coverage and overhead numbers in §4 come from
  injected bugs and generated edits; the repair loop has run only with a scripted model (Task 4 above) and the
  earlier GLM pilot.
- **One real miss in the final evaluation.** `layernorm_txl_ref` with `x[r, 0]` → `x[r, 0 + 1]` reads one element
  past `x` on the last row when N = 1 (outputs stay within tolerance). The gate would catch it if its shapes
  included N = 1 (they are T+1 and 1000). Not changed: it was found on the fresh set.
- **VOLTA:**
  - the ≥ 2^127 constant rewrite is applied unconditionally; it is exact only when the constant is a max / min
    identity (a kernel computing `x + 3.4e38` would be modelled wrongly);
  - `softmax_txl_bug_tail` at N = 100 is UNSUPPORTED;
  - tile changes in GEMM are undecided (block 0 covers another region).
- **Held-out is partly inspected.** The size-regime rule and one IR-pairing fix came from looking at held-out misses,
  so the held-out set is not fully held out. The fresh set (tirx LayerNorm / GEMM) had nothing tuned on it; the one
  harness fix found there (misaligned pointers, §4) only removes false catches.
- **Generated edits are single, local changes.** Real agent revisions are often 10–50-line rewrites; the classifier
  then labels most categories, so the tests are broad rather than targeted.
- **Cost of the IR analysis:** about 8 s per revision, mostly the tvm import in a fresh child process; a persistent
  worker would remove most of it.
- **LLM second opinion** for the classifier: not yet tried with real calls.
- **GEMM leaves the most room for the agent.** `gemm_txl_opt` reaches 0.29× PyTorch (FP32 SIMT, no tensor cores:
  TF32 fails the fp32 tolerance).

---

## 3. Interfaces

### Workloads are data
`workloads/<op>/manifest.json` describes:
- shape keys;
- launch kind / block / tiles;
- the fixed test suite;
- numerical distributions;
- VOLTA default / deep shapes and a size cap;
- the tirx factory arguments;
- the declared shape domain and the benchmark shape;
- the reference files;
- the variants, including injected bugs with their category and expected VOLTA verdict.

**Shape domain (CAKE contract).** The optional `domain` key fixes the valid shape range before tuning, e.g.
`{"rows": [1, 4096], "N": [1, 16384]}` (gemm: M, N, K in [1, 4096]). The kernel must be correct on all of it and is
timed at the one `perf.shape` inside it; generated tests, VOLTA shapes, size-regime and LLM-suggested sizes never
leave it, `akc bench` rejects a shape outside it, and the agent's prompt states it (`agent/pilot.py: contract_text`).
Without `domain`, nothing is limited, as before.

Adding a workload needs **three things**. Each optional add-on is switched on by adding its piece; without it, that
feature reports "not set up" instead of failing.

**Required:**
1. a manifest with `op`, `description`, `shape_keys`, `launch` (`row` or `tile2d`), `fixed_suite` and `variants`;
2. the kernel files;
3. an `OpSpec` in `akc/task/reference.py` with `make_inputs`, `golden`, `out_shape`, `kernel_args`, `atol` and
   `rtol`.

**Optional add-ons:**

| add-on | what to add |
|---|---|
| VOLTA | a reference kernel in the candidate's language (`references.tirx`; legacy: `references.cuda`) and a `volta` block (default / deep shape, size cap) in the manifest. The VOLTA memory layout is derived from the `OpSpec`; the output array takes its name from the manifest `signature` |
| speed | `perf.shape` in the manifest, plus `flops`, `bytes` and `torch_impl` in the `OpSpec` |
| shape domain | `domain` in the manifest: `{shape key: [lo, hi]}` (inclusive) |
| tirx-lite | a `tirx` block in the manifest and a tirx reference kernel |
| better change-aware tests | `tiles` and `numerical_dists` in the manifest (defaults: the launch block or tile, and large / negative large / tiny / mixed values) |
| evaluation | bug variants with `bug` and `expect_volta` |
| a launch other than row or tile2d | a `launch` function in the `OpSpec` |

**Limit:** the change-aware test generator has built-in size patterns for two shape families only:
- `rows, N` (row-wise ops);
- `M, N, K` (tiled matrix ops).

An operator with other shape keys, for example attention's `B, H, S, D`, also needs size patterns added to
`generate.py`.

Kernel conventions (CUDA):
- entries are `extern "C" __global__`;
- write exp as `EXP(x)` (from `workloads/common/common.cuh`);
- control flow must not depend on data values.

### Validator contract (`akc/validation/interface.py`)
Every validator implements `validate(cand: Candidate, ctx: ValidationContext) -> ValidationReport`.

| type | fields |
|---|---|
| `Candidate` | `workload`, `lang` ("cuda" / "tirx"), `source`, `entry`, `block`, `parent` (the last **accepted** version), `label`, `path`; `Candidate.from_file(workload, path, **kw)` |
| `ValidationContext` | `config`, `test_cases`, `volta_shapes`, `device` ("auto" / "gpu" / "cpu"), `time_budget_s`, `extra` |
| `Finding` | `kind` (compile_error, mismatch, nonfinite, race, not_equivalent, out_of_bounds, unwritten, nondeterministic, unsupported, timeout, …), `severity` (error / warning / info), `validator`, `message`, `evidence`, `location` |
| `ValidationReport` | `validator`, `passed` (no error findings), `decided` (→ verification coverage), `findings`, `seconds` (→ validation overhead), `artifacts` |

### Main entry points
| function | returns | notes |
|---|---|---|
| `validation.pipeline.run_pipeline(cand, config="C", ctx=None, cache=None, skip=())` | `PipelineResult(config, reports, seconds, stopped_after, skipped, cached)` with `.passed`, `.findings`, `.decided_by`, `.per_validator`, `.validation_seconds` | A0: fixed tests; A: gate + fixed tests; B: A + VOLTA; C: B + change-aware tests, with VOLTA at the change-aware shapes. Stops at a compile error; stops at the first failure unless `ctx.extra["run_all"]`. `cache` (one dict per candidate) reuses a validator's report across configurations |
| `validation.compiler_checks.CompilerChecksValidator().validate(cand, ctx)` | `ValidationReport("compiler_checks")` | the gate (D2): tirx-lite static checks, then compute-sanitizer (GPU) or NumSim racecheck / synccheck (CPU). Legacy CUDA: one info finding |
| `validation.changeaware.plan(workload, old_src, new_src, lang=None, budget=16, llm=None)` | `ChangePlan(hunks, labels, cats, tests, volta, tiles, llm)` | analysis only; nothing is run. `llm`: an optional `LlmClassifier` |
| `perf.measure(cand, shape=None)` | `PerfReport`: `time_ms`, `speedup_vs_torch`, `achieved_gbps` / `achieved_tflops`, `pct_peak_*`, `bound`, `floor_ms`, `headroom`, or `measured=False` with a `reason` | GPU only. It re-checks correctness at the benchmark shape. Call it only for candidates that passed; it is identical in A/B/C |
| `feedback.render_observation(reports, perf=None, best=None)` | text for the agent | validation findings first, then speed |
| `llm.lumen.LumenClient(model="glm-5.3").chat(messages)` | `ChatResult` | budget guard, retries, JSONL log; uses your Lumen key |

Example: one agent step.
```python
from akc.validation.interface import Candidate, ValidationContext
from akc.validation.pipeline import run_pipeline
from akc.perf import measure
from akc.feedback.render import render_observation

parent = Candidate.from_file("softmax", "workloads/softmax/tirx/softmax_txl_ref.py")   # .py -> tirx-lite
cand = Candidate.from_file("softmax", "my_softmax.py", parent=parent, label="r2")
res = run_pipeline(cand, "C", ValidationContext(device="auto"))
perf = measure(cand) if res.passed else None          # speed only for correct kernels (needs a GPU)
print(render_observation(res.reports, perf))
```

### Command line (after `source env.sh`)
| command | what it does | needs |
|---|---|---|
| `akc validate --workload W --kernel F --config A0\|A\|B\|C [--run-all] [--perf] [--llm-classify] [--json out.json]` | prints exactly what the agent would see: validation, then speed | CPU (VOLTA, NumSim); GPU for tests, sanitizer and speed; Lumen key with `--llm-classify` |
| `akc hottest --workload W --new F [--old F] [--llm-classify]` | what changed, the targeted tests, the problems found | CPU / GPU |
| `akc check --workload W [--lang cuda\|tirx] [--variants a,b] [--extra L=F] [--tests t.json]` | runs tests on variants or extra kernel files | GPU (tirx also CPU via NumSim) |
| `akc bench --workload W [--lang tirx] [--feedback]` | time, ×PyTorch, % of peak, practical floor | GPU |
| `akc volta --workload W [--lang cuda\|tirx]` | VOLTA on every candidate variant, checked against the manifest's expectations | CPU |
| `akc eval-bugs [--lang tirx] [--device gpu] [--configs A0,A,B,C]` | configurations A0/A/B/C on the injected bugs, through `run_pipeline` (run_all), caught / seconds per validator | GPU for the tests and the sanitizer |
| `akc mutants [--set bases\|heldout\|fresh\|all] [--lang tirx] [--llm-classify] [--detect --device gpu [--volta [--volta-workers N]]]` | mutation benchmark: is each generated edit classified right? (with `--detect`: caught by A0 / A / (B) / C?). `--volta-workers`: VOLTA in N parallel CPU processes while the GPU validators run serially (`evaluation/volta_pool.py`) | CPU; GPU with `--detect` |
| `akc tirx-gpu` | tirx-lite kernels on the GPU (correctness + speed) | GPU |
| `akc agent --workload W [--config C] [--seed F] [--optimize] [--max-iters 10] [--max-coins 2] [--dry-run]` | the repair loop (Task 4): model → configuration → feedback, every iteration logged under `results/agent/` | Lumen key (coins); CPU or GPU for validation, GPU for speed |
| `akc pilot`, `akc lumen models\|usage\|chat` | GLM pilot loop / Lumen client | Lumen key |
| `sbatch scripts/slurm/smoke.sbatch` | full check on an A100 (~10 min): unit tests, A/B/C, speed, observations | Slurm (`bidr-delta-gpu`, `gpuA100x4-interactive`) |

---

## 4. Tests run

### Unit tests
Run with `python -m unittest discover -s tests`.

| file | tests | covers |
|---|---|---|
| `tests/validation/test_interface.py` | 15 | the contract types; pipelines (B misses / C catches the tail bug, compile errors stop the pipeline); parent policy; `akc hottest`; a tirx bug caught on the CPU (NumSim) |
| `tests/validation/test_compiler_checks.py` | 28 | the gate: sanitizer log parser on saved logs; TIRx helper reports mapped to kernel lines; findings and feedback text; gate shapes; pipeline order, stop rules, `run_all`, cache and `skip`; static checks on the tirx refs / opts / bugs and an IR-contract violation; 3 GPU tests |
| `tests/validation/test_changeaware_ir.py` | 13 | IR change analysis: offbyone → indexing, tail only at non-multiples of 128, nosync → one sync hunk "×14" (and C's VOLTA shapes for tirx), nomax → numerical, rename → no hunks, AST fallback, indexing → tiling |
| `tests/validation/test_changeaware.py` | 45 | diff (reformat / rename = no change), the text classifier, the generator (floor tests, size regimes, shape domain, no misaligned pointers for tirx-lite), VOLTA shapes, the LLM option with a fake client |
| `tests/validation/test_volta.py` | 40 | VOLTA verdicts and status mapping; memory layouts; the PTX build; the tirx-lite path (rewrites, entry parsing, `analyze` for out-of-bounds) |
| `tests/validation/test_tirx_kernels.py` | 13 | tirx-lite LayerNorm / GEMM: trace and compile, NumSim, VOLTA verdicts on refs / opts / bugs, GPU runs |
| `tests/validation/test_runners.py` | 22 | input distributions; PyTorch references; output checks (mismatch, unwritten, canary, NaN); tolerances; GPU runs |
| `tests/task/test_workload.py` | 12 | manifests, the declared shape domain (`in_domain`, `clip`), derived launch and VOLTA layout |
| `tests/task/test_outlier.py` | 7 | outlier inputs: every residue mod P, alternating row scales, the magnitude ladder, never the floor distribution |
| `tests/perf/test_perf.py` | 16 | roofline arithmetic, speed feedback text, the benchmark shape, measurement (2 GPU tests) |
| `tests/llm/test_lumen.py` | 8 | the Lumen client against a fake server (no coins) |
| `tests/evaluation/test_bugset.py` | 3 | the bug-set evaluation's parent choice; every configuration goes through `run_pipeline` |
| `tests/evaluation/test_mutants.py` | 7 | the mutation generator, the fresh set, no edits of a float's exponent, a classifier regression guard |
| `tests/evaluation/test_hard_edits.py` | 3 | the 34 hard edits: re-selected from the full run (A 19, B 29, C 33), the 11 no-change edits never caught, the generator still produces every key |
| `tests/evaluation/test_volta_pool.py` | 5 | parallel VOLTA for the evaluation: merged report = serial report, cache routing, shape superset, one real run |
| `tests/agent/test_loop.py` | 10 | the repair loop with a fake model and validators (§2 Task 4) |
| **total** | **247** | Login node, 2026-11-18: **247 run, all pass, 11 skipped** (GPU-only, or a tool missing). A100 (jobs 24061708 / 24061793): the gate, interface and bug-set tests, 46 run, all pass |

### Experiments (A100-SXM4-40GB; GPU runs logged in `notes/gpu_results.md`)
**Setup from scratch:** a fresh clone, `setup_env.sh --tests`, took 8 min 52 s, and all tests passed (86 tests at the time).

#### Main experiment: configurations A0 / A / B / C on tirx-lite (Task 5)
2026-11-17, jobs 24062207, 24062451, 24064163 and 24078424 (2 h 10 min of A100 time in all).

**Setup.**
- Every candidate runs through `run_pipeline`, exactly as in the agent loop, with `run_all` so that every
  validator's catches can be counted. The parent is the correct version the candidate was made from.
- A candidate is **caught** by a configuration if any of its validators reports an error.
- Candidates (the detection workload; every one is listed and explained in `notes/detection_workload.md`):
  - the 11 injected tirx-lite bugs;
  - the **34 hard edits**: generated single edits (`akc mutants --detect --volta`) that change the kernel's
    behaviour and that plain fixed tests miss, from three sets of correct kernels:
    - **tuning set:** `softmax_txl_ref`; the classifier rules were revised while looking at it;
    - **held out:** 4 GLM-written tirx softmax kernels, only partly held out (see Open items);
    - **fresh:** `layernorm_txl_{ref,opt}` and `gemm_txl_{ref,opt}`; nothing was tuned on them;
  - 27 no-op edits (renames), to count false alarms.
- **How the 34 were chosen; the archive.** This run used all 164 generated edits. 119 of them are caught by plain
  fixed tests in every configuration (36 do not compile, 83 give visibly wrong output), so they cannot separate A, B
  and C. 11 cannot change any output (listed below). These 130 are archived: `akc mutants --detect` now skips them
  unless `--archived` is given (`akc/evaluation/hard_edits.py`). `python -m akc.evaluation.hard_edits` rebuilds
  `results/eval/hard_edits.json` from this run.
- VOLTA ran in parallel CPU worker processes (`--volta-workers`) while the GPU validators ran one at a time.
  The held-out and fresh sets ran as two streams on one GPU, so their timings are not used below.

**Detection** (caught / candidates):

| candidates | A0: fixed tests | A: + gate | B: + VOLTA | C: + change-aware tests |
|---|---|---|---|---|
| 11 injected bugs | 3 | 6 | 6 | **11** |
| hard edits, tuning set (6) | 0 | 5 | 5 | **6** |
| hard edits, held out (13) | 0 | 4 | 10 | **13** |
| hard edits, fresh (15) | 0 | 10 | 14 | **14** |
| **all 34 hard edits** | 0 | 19 | 29 | **33** |
| false alarms on 27 no-op edits | 0 | 0 | 0 | 0 |

A0 catches none of the hard edits by construction.

**What each step adds:**
- **The gate (A0 → A)**, 3 bugs and 19 hard edits:
  - races: racecheck finds `layernorm_txl_bug_nosync`, which every GPU test misses, and 6 edits (2 deleted
    barriers, 4 `<` → `<=` flips, some of which also read out of bounds);
  - out-of-bounds and uninitialized reads whose outputs still look right: memcheck and initcheck, mostly `<` → `<=`
    flips in guards and loop bounds, plus a halved block size.
- **VOLTA at the default size (A → B)**, 0 bugs and 10 hard edits:
  - 7 are `if (lane == 0)` / `if (tid == 0)` negations. Every thread then writes the same value to the same
    address. That is a race by the memory model, but the outputs never change, and racecheck does not flag it;
  - 2 halve a shared array, so accesses go past the shared memory: outputs happen to stay right;
  - 1 is a value bug (`+` → `-` in `layernorm_txl_opt`).
- **Change-aware tests, with VOLTA at their sizes (B → C)**, 5 bugs and 4 hard edits:
  - the bugs are the 3 tail bugs (VOLTA also catches them, at the change-aware sizes), `nomax` and `onepass`;
  - the edits are float-only failures: max → min, a max initialized to 0 instead of -FLT_MAX (twice), and a warp max
    that skips half the lanes. All of them give inf / NaN on large or outlier inputs, which VOLTA's real-number model
    cannot see.
- **The two mechanisms are complementary.** On the injected bugs, change-aware tests catch all 11 and VOLTA-C
  catches 9 (all but the two numerical bugs). On the hard edits, VOLTA alone catches 9 and the change-aware tests
  alone catch 4.

**What C misses: 1 hard edit**, an out-of-bounds read at N = 1 whose outputs stay within tolerance (see Open items).

**The 11 archived edits that cannot change any output** (no configuration flags them, as expected):
- 9 change a block size, tile size or unroll factor in a kernel that derives everything else from it. VOLTA proves
  the 5 GLM ones equivalent. It cannot decide the 4 GEMM ones, because block 0 then covers another part of the
  output; we judged them harmless by reading the code;
- 2 pad the GEMM shared tiles with 1.0 instead of 0.0. The padding is always multiplied by the other operand's zero
  padding. VOLTA proves both equivalent.

**Verification coverage** (VOLTA decided / VOLTA ran):
- injected bugs: B 11/11, C 10/11 (softmax tail at N = 100 is UNSUPPORTED);
- hard edits: 34/34 in both B and C;
- all 164 generated edits of the full run: 114/132 (86%) in both B and C. VOLTA does not run on edits that fail to
  compile. The undecided ones (tile changes in GEMM, VOLTA-unsupported constructs) are all archived edits.

**Overhead** (seconds of validation per candidate, from the single-stream runs; each validator's own time):

| candidates | A0 | A | B | C |
|---|---|---|---|---|
| 11 injected bugs (mean) | 9.3 | 19.5 | 22.4 | 49.1 |
| 24 tuning-set edits (mean) | 8.7 | 18.6 | 22.4 | 50.2 |

Per validator (median):
- gate 10–11 s;
- fixed tests 9 s;
- change-aware tests 19–21 s, including about 8 s of IR analysis;
- VOLTA about 3 s at the default size and 9–13 s at the change-aware sizes.

**A test-harness bug the run found (fixed).**
- The change-aware "memory" tests passed tensors misaligned by one element. A TIRx-compiled function checks every
  argument's alignment and refuses such a call (`ValueError: Misaligned Tensor data on argument #0`, also on the
  correct reference; job 24078424). So every edit labelled "memory" was "caught".
- 7 equivalent edits were counted as caught by C this way.
- The generator no longer makes misaligned tests for tirx-lite. The 27 edits labelled "memory" were re-run; exactly
  those 7 changed. The table above is after the fix.
- Before the fix: `results/eval/mutants_detect_tirx_{heldout,fresh}_before_misalign_fix.json`.

Results: `results/eval/bugset_tirx_gpu_20261117-053113.json` (injected bugs), `results/eval/hard_edits.json` (the
34 hard edits), `results/eval/mutants_detect_tirx_{bases,heldout,fresh}.json` (the full run, including the archived
edits).

#### Compiler-checks gate (D2) on tirx-lite
A100, job 24061793 (`scripts/experiments/gate_tirx.py`, `results/eval/gate_tirx.json`). "Caught" means the tool
reported a problem in our kernel. This was the first time compute-sanitizer ran on TIRx-generated code.

| kernels | memcheck | racecheck | initcheck | synccheck | gate (any) |
|---|---|---|---|---|---|
| 5 tirx refs / opts (false alarms) | 0/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| 4 GLM tirx kernels (false alarms) | 0/4 | 0/4 | 0/4 | 0/4 | **0/4** |
| 11 injected tirx bugs | 3/11 | 3/11 | 0/11 | 0/11 | **6/11** |
| 8 tirx edits VOLTA calls RACE / OUT_OF_BOUNDS | 2/8 | 3/8 | 2/8 | 0/8 | **4/8** |
| 6 tirx edits VOLTA calls EQUIV | 1/6 | 0/6 | 0/6 | 0/6 | 1/6 |

- **Injected bugs.** racecheck catches the 3 nosync bugs. memcheck catches the 2 off-by-ones and the gemm guard.
  The 5 value bugs need the tests or VOLTA.
- **Beyond VOLTA.** `glm_txl_C_r2` with `[r, col]` → `[r, col + 1]` reads one element past `x` on GLM's large-row
  path (N = 12285, a gate shape from the size-regime rule). VOLTA said EQUIV: it checks only N ≤ 584.
- **Locations.** Every report points at a kernel-body line of the generated CUDA (both accesses of a race), after
  mapping TIRx's inlined helpers back to their call sites.
- **Cost per revision** (median of 34): **10.9 s** (max 12.7 s).
  - Static checks take 7.8 s: tvm import 1.7 s, two sm_80 compiles 2.8 s.
  - The 4 tools take 2.1 s in parallel (4.5–4.7 s one after another, with the same verdicts).
- **CPU-only path** (NumSim): the same race and out-of-bounds verdicts on GEMM, 12 s per kernel.

#### Change classifier: IR vs AST on the tirx-lite edits
`akc mutants --classifier ir|ast`, CPU. 106 tirx edits: 24 on the reference, 82 on the 4 held-out GLM tirx kernels.
The fresh set (85 edits) was classified with the IR only: 73/73 right category, 0/12 no-op edits labelled.

| kernels | classifier | right category | wrong | no category | no-op edits labelled | AST fallback | `ir_blind` |
|---|---|---|---|---|---|---|---|
| tirx reference (tuning) | AST | 21/21 | 0 | 0 | 0/3 | – | – |
| tirx reference (tuning) | IR | 21/21 | 0 | 0 | 0/3 | 2 | 0 |
| 4 GLM tirx (held out) | AST | 70/70 | 0 | 0 | 0/12 | – | – |
| 4 GLM tirx (held out) | IR | 70/70 | 0 | 0 | 0/12 | 8 | 0 |

- **Agreement.** Every edit gets the same verdict from both classifiers.
- **Category sets** differ on 36 of 106 edits, the first category on 16. The IR adds categories more often than it
  drops them, mostly numerical from the reduction rule. Examples:
  - a guard or loop bound inside a max / sum loop is indexing (or tiling) + numerical;
  - a shuffle offset `16 → 8` in a warp max is sync + numerical (AST: indexing).
- **Cost of the broader labels:** 1409 generated tests vs 1027 with the AST (+37%).
- **Fallbacks:** the AST fallbacks are `float32 → float16` edits that TIRx refuses to trace.
- **Analysis time:** about 7.7 s per revision on the IR path, mostly the tvm import; the AST path takes 0.01 s.
- **Not fully held out:** before one pairing fix, the IR scored 69/70 on the held-out set.

Results: `results/eval/mutants_rules_ir_{bases,heldout,fresh}.json`.

#### Speed
rows = 4096, N = 1024; GEMM 1024³; median of 50 launches, L2 flushed.

| kernel (tirx-lite) | time | × PyTorch | throughput |
|---|---|---|---|
| softmax_txl_ref | 0.033 ms | 1.08× | 65% of peak bandwidth |
| layernorm_txl_opt | 0.035 ms | 1.07× | 62% of peak bandwidth |
| gemm_txl_opt | 0.47 ms | 0.29× | 4.6 TFLOP/s, 23% of FP32 peak (PyTorch: 64%) |
| layernorm_txl_ref / gemm_txl_ref (simple references) | 0.30 / 1.09 ms | 0.13× / 0.12× | |

A device copy of the same bytes reaches about 66–68% of the datasheet bandwidth. So softmax and LayerNorm are near
the practical limit, and GEMM is where an optimizing agent can gain the most.

#### GLM-5.3 as the kernel writer (pilot)
It wrote correct softmax kernels in tirx-lite (4/4 runs) and in CUDA (4/4) within 3 rounds. It also produced two
real, non-injected bugs:
- a shared-memory race: caught only by VOLTA; every GPU test passed;
- a dropped term in an unrolled loop that runs only for N > 384: caught only by the GPU tests at large N. After this
  we added a "deep" VOLTA size.

#### Problems the GPU runs found (all fixed and re-verified)
- **A false alarm.** The LayerNorm "offset" test (mean 1e4) rejected the correct `layernorm_opt`: rounding the row
  mean to float32 is unavoidable. Fixed with a condition-aware tolerance.
- **A kernel that never stops.** `off > 0` → `off >= 0` hung the runner for 900 s. Now there is a 180 s limit.
- **Broken base kernels counted.** 3 held-out GLM CUDA kernels are wrong by themselves, so all their edits counted as
  "caught". Such bases are now excluded.
- **A path no test reached.** GLM's large-row code path (N > 8192 in CUDA, n > 12284 in tirx) was never run. This
  led to the size-regime tests (§2).
- **memcheck hung** with `--destroy-on-device-error kernel` after reporting an out-of-bounds read. It now uses
  `context`.
- **Misaligned-pointer tests on tirx-lite** (above).

#### Legacy: hand-written CUDA results
These came before the switch to tirx-lite as the only agent language. They are kept for reference and are not
part of configurations A0/A/B/C. Here "A" is today's A0 (fixed tests only), and "C" is fixed + change-aware tests.
- **VOLTA feasibility (CPU).** 32/32 verdicts matched expectations on the CUDA softmax, GEMM and LayerNorm:
  - optimized kernels: EQUIV;
  - missing barriers: RACE;
  - off-by-one: NOT_EQUIV;
  - numerical bugs: EQUIV, the known blind spot.
- **11 injected CUDA bugs:** A 4/11, B 5/11, C 11/11.
- **Classifier on 387 CUDA + tirx edits** (`results/eval/mutants_rules_{before,after}_*.json`):
  - before the rule revision: 67% right on the tuning set, 62% held out;
  - after: 198/198 and 189/189, with 0 no-op edits labelled.
- **Detection on the edits** (A100, jobs 24051467 / 24052255 / 24052791; `results/eval/mutants_detect_{bases,heldout}.json`):
  - tuning set: A 83/136, C 105/136;
  - held out: A 45/67, C 53/67;
  - no false alarms.

  Of C's 45 misses:
  - 14 CUDA races / shared out-of-bounds, all caught by VOLTA;
  - 8 float-only bugs: a warp max that skips half the lanes. These motivated the outlier input family;
  - 9 equivalent edits;
  - 14 tirx edits (VOLTA did not run on tirx then).

  Details: `notes/build_log.md` §8c–8e.
- **Speed:**
  - softmax_opt 0.038 ms (0.96×);
  - best GLM CUDA softmax 0.034 ms (1.07×);
  - layernorm_opt 0.038 ms (1.12×);
  - gemm_opt 0.61 ms (0.28×).

### Where results are
| results | location |
|---|---|
| Slurm logs | `logs/slurm/` |
| bug-set results | `results/eval/` |
| speed | `results/perf/` |
| VOLTA runs | `results/volta/` |
| agent runs (Task 4) | `results/agent/` |
| GLM pilot | `results/pilot/` |
| change-aware test lists | `results/change_aware/` |

## Layout
```
akc/            the system: task, llm, agent, backend, validation, perf, feedback, evaluation
workloads/      one folder per operator: manifest.json + cuda/ (tirx/) kernels and injected bugs
tests/          unit + integration tests (mirrors akc/)      scripts/slurm/  GPU jobs
docs/           architecture, setup                         notes/          dated decisions and experiment logs
setup/          exact package lock                          results/        experiment outputs (JSON)
third_party/    VOLTA, TIRx-harness, tirx-kernels, Lumen docs (fetched by setup_env.sh, not committed)
```
