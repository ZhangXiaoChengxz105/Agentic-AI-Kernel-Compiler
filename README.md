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

Status (2026-11-16): **Tasks 1–2 done** (environment, baseline harness, change-aware testing; details in §2).
2026-11-17: the compiler-checks gate (D2) and configurations A0/A/B/C are in place (§2 D2, results in §4).
Tasks 3–5 have groundwork in place (a VOLTA validator, the A/B/C pipeline, feedback text, a GLM pilot loop,
bug-set evaluation); their owners will extend it.

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
LLM calls (`akc pilot`, `akc lumen`) use your own Lumen key: put `export LUMEN_API_KEY=...` in
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

We also support plain **CUDA**. Every component handles both languages; VOLTA reads tirx-lite through rewrites of
its generated code (see Open items). Which language the agent writes in the final experiments is still a team decision.

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

  Configuration A's fixed test suite is in `akc/validation/fixed_tests/`. tirx-lite kernels can also run on the CPU
  through NumSim.
- **Speed measurement.** `akc/perf/` times a kernel with CUDA events: warmup, L2 cache flushed before every launch,
  median of 50 runs. It compares the time with three things:
  - PyTorch;
  - the A100's peak bandwidth and FLOP/s;
  - a device copy of the same bytes, which is the practical floor.
- **VOLTA check.** VOLTA can analyze all three CUDA kernels: 32/32 verdicts matched our expectations
  (`notes/volta_scope.md`). Its limits:
  - real-number arithmetic;
  - only thread block 0;
  - exact sizes only;
  - `exp` must be written as `EXP()`.
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
   - memory → odd sizes and pointers misaligned by one element;
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

Example: the change-aware report (`akc hottest`) for `softmax_bug_tail`. That bug changes the loop bound from
`i < N` to `i < (N / BLOCK) * BLOCK`, so the last N % 128 elements are dropped.
- **Classification:** tiling + indexing (lines 20 and 32).
- **Tests:** 12 targeted, at N = 1, 127, 129, 255, 257 and 1000, plus 1 floor test with large values. On the A100,
  all 12 targeted tests fail (run 24032011).
- **VOLTA:** sizes N = 256, 584, 200 and 100. VOLTA proves the kernel wrong at N = 200 and 584. At the default
  N = 256, a multiple of 128, it says "equivalent", which is why configuration B misses this bug.

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

### Groundwork beyond Tasks 1–2 (for Tasks 3–5)
- **VOLTA validator** (`akc/validation/volta/`) with actionable hints.
- **Configurations A0/A/B/C** (`akc/validation/pipeline.py`).
- **Feedback text** for the agent (`akc/feedback/render.py`).
- **Speed objective** (`akc/perf/`).
- **Bug-set evaluation** (`akc/evaluation/`).
- **GLM-5.3 pilot loop** (`akc/agent/pilot.py`).
- **Injected bugs** (off-by-one, missing barrier, tail/tiling, numerical): 11 in tirx-lite
  (`workloads/*/tirx/bugs/`: softmax 4, LayerNorm 4, GEMM 3) and the 11 legacy CUDA originals (`workloads/*/cuda/bugs/`).

### Open items
- **VOLTA in configuration C gets no change-aware shapes for tirx-lite.** `changeaware.plan()` returns
  `volta_shapes(...) if lang == "cuda" else []` (since the restructure), so the pipeline hands VOLTA an empty list
  and it falls back to the default shape: for tirx, VOLTA-C = VOLTA-B. Found in the D2 evaluation (job 24061793);
  not fixed in D2. The likely fix is `volta_shapes(workload, cats, tiles)` for both languages, then re-running
  `akc eval-bugs --lang tirx`.
- **VOLTA on tirx-lite (Task 3): works for softmax; the rest is open.** `VoltaValidator` traces a tirx-lite candidate
  and `softmax_txl_ref` at each VOLTA size, rewrites TIRx's generated CUDA so that VOLTA can read it, and runs
  `volta compare` (`akc/validation/volta/tirx.py`; details in `notes/volta_scope.md`).
  - **What works:**
    - All 23 tirx kernels we tried are decided at N = 256, 584 and 200: 69 of 69 runs, coverage 100%. They are the
      reference, 4 injected bugs, 4 GLM kernels, and 14 mutants that the tests missed.
    - Verdicts match the CUDA counterparts.
    - VOLTA catches 8 of the 14 missed mutants, including all 5 races.
  - **Not exact:** two of the five rewrites are modelling choices:
    - `ex2.approx(t)` is read as exp(t / log2 e);
    - constants of magnitude 2^127 or more (-3.4e38, -FLT_MAX as the identity of a max) become -inf.
  - **Not covered:**
    - data-dependent control flow;
    - partial-CTA barriers and async copies;
    - transcendental functions other than exp;
    - code paths that only run above the VOLTA sizes.
  - **Not re-run:** the tirx numbers for `eval-bugs` and mutant detection in §4 predate this change. B and C now
    include VOLTA for tirx.
- **tirx-lite LayerNorm and GEMM (Task 1, 2026-11-17)** are now the mutation benchmark's `fresh` set (D2, below).
  Classification on the CPU: 73/73 semantic edits get a right category, 0/12 no-op edits are labelled
  (`results/eval/mutants_rules_ir_fresh.json`). Detection on the GPU (`--set fresh --detect`) has not been run.
  `akc check` and `akc volta` take `--lang tirx`.
- **Detection is measured on our own bugs only.**
  - The 11/11 bug detection uses bugs written by the same person as the classifier.
  - The classifier itself is now measured on mechanically generated edits, including edits of GLM-written kernels
    (§4).
  - Detection on those generated edits is measured too: tests A 61% → C 77% on the tuning set, 67% → 79% held
    out. Most remaining misses are races (VOLTA's job) or edits that change nothing (§4).
  - Still missing:
    - a detection re-run with the new outlier input family (§2 Task 2). It targets the 8 CUDA edits whose max
      reduction skips half the warp: they give NaN only on rows with one large value. It has not been run on the
      GPU yet;
    - the LLM option, measured with real calls.
- **GPU tests are weak at races.** `layernorm_bug_nosync` is caught only by VOLTA.
- **Our GEMM leaves the most room for the agent.** It reaches 0.28× PyTorch, and TF32 tensor cores cannot be used
  because they fail the fp32 tolerance.

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
| VOLTA | a CUDA reference kernel (`references.cuda`) and a `volta` block (default / deep shape, size cap) in the manifest. The VOLTA memory layout is derived from the `OpSpec`; the output array takes its name from the manifest `signature` |
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

parent = Candidate.from_file("softmax", "workloads/softmax/cuda/softmax_opt.cu", entry="softmax_opt")
cand = Candidate.from_file("softmax", "my_softmax.cu", parent=parent, label="r2")
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
| `akc mutants [--set bases\|heldout\|fresh\|all] [--lang tirx] [--llm-classify] [--detect --device gpu [--volta]]` | mutation benchmark: is each generated edit classified right? (with `--detect`: caught by A0 / A / (B) / C?) | CPU; GPU with `--detect` |
| `akc tirx-gpu` | tirx-lite kernels on the GPU (correctness + speed) | GPU |
| `akc pilot`, `akc lumen models\|usage\|chat` | GLM pilot loop / Lumen client | Lumen key |
| `sbatch scripts/slurm/smoke.sbatch` | full check on an A100 (~10 min): unit tests, A/B/C, speed, observations | Slurm (`bidr-delta-gpu`, `gpuA100x4-interactive`) |

---

## 4. Tests run

### Unit tests
Run with `python -m unittest discover -s tests`.

| file | tests | covers |
|---|---|---|
| `tests/validation/test_interface.py` | 15 | the contract types; pipelines A/B/C (B misses / C catches the tail bug, compile errors stop the pipeline); parent policy; `akc hottest`; a tirx bug caught on the CPU (NumSim) |
| `tests/validation/test_runners.py` | 22 | input distributions; PyTorch references; output checks (mismatch, unwritten, canary, NaN); tolerances, including the LayerNorm offset false-alarm regression; GPU runs of references, optimized kernels and bugs |
| `tests/validation/test_volta.py` | 21 | VOLTA verdicts (EQUIV, NOT_EQUIV, RACE, unsupported, timeout); status mapping; memory layouts; the VOLTA PTX build (`__symexpf`, no `.loc`) |
| `tests/validation/test_changeaware.py` | 37 | diff (incl. reformat / rename = no change), classifier (incl. loop steps, while bounds, float vs. index arithmetic, warp shuffles), generator (incl. floor tests and size-regime thresholds), VOLTA shapes, the LLM option with a fake client |
| `tests/task/test_workload.py` | 5 | optional manifest / OpSpec parts, derived launch, derived VOLTA layout = the old hand-written one |
| `tests/perf/test_perf.py` | 15 | roofline arithmetic, speed feedback text, measurement (2 GPU tests) |
| `tests/llm/test_lumen.py` | 8 | the Lumen client against a fake server (no coins) |
| `tests/evaluation/test_bugset.py` | 3 | which parent the bug-set evaluation uses (CUDA: the optimized kernel; tirx: the reference); every configuration goes through `run_pipeline` with `run_all` and per-validator numbers |
| `tests/evaluation/test_mutants.py` | 7 | the mutation generator (incl. which edits cannot change outputs), the fresh set loads, extended tirx operators only where asked, no edits of a float's exponent, plus a regression guard on the classifier |
| `tests/validation/test_changeaware_ir.py` | 13 | IR change analysis: offbyone → indexing at L51, tail only at non-multiples of 128, nosync → one sync hunk "×14", nomax → numerical, re-trace / rename → no hunks, a float16 mutant that does not trace → AST fallback, swapped operands and another buffer detected, indexing → tiling, same categories → same tests (8 need `tvm_tirx`) |
| `tests/task/test_outlier.py` | 7 | outlier inputs: every residue mod P, alternating row scales, the magnitude ladder, deterministic, never the floor distribution, the generator's `outlier@T` |
| `tests/validation/test_compiler_checks.py` | 28 | the gate (D2): sanitizer log parser on saved logs (and every trial log); TIRx helper reports mapped to kernel lines; report / NumSim / static results → finding kinds; feedback text; gate shapes; pipeline order, stop rules, `run_all`, cache and `skip` with fake validators; A0 = the old A; static checks pass on the 5 tirx refs / opts and 11 bugs and fail on an IR-contract violation at its line (with and without `check_ir`); 3 GPU tests (nosync → race, gemm guard → out_of_bounds, references clean) |
| **total** | **129** | Login node (2026-11-16): **129 run, all pass, 6 skipped** (GPU-only). A100 (job 24048511, before the last 3 tests were added): 126 run, all pass, 2 skipped |

Login node, 2026-11-17, after the IR change analysis (D3); other files' counts have grown since the table: 196 run, all pass, 8 skipped (GPU-only, or `tvm_tirx` / VOLTA missing).

Login node, 2026-11-17, after D2: 228 run, all pass, 11 skipped (GPU-only, or a tool missing). A100 (jobs 24061708
and 24061793): `test_compiler_checks`, `test_interface` and `test_bugset` 46 run, all pass, 1 skipped; the 3 GPU
gate tests pass again after the memcheck change.

### Experiments (A100-SXM4-40GB; GPU runs logged in `notes/gpu_results.md`)
**Setup from scratch:** a fresh clone, `setup_env.sh --tests`, took 8 min 52 s, and all tests passed (86 tests at the time).

**VOLTA feasibility (CPU).** 32/32 verdicts matched expectations on softmax, GEMM and LayerNorm:
- optimized kernels: EQUIV;
- missing barriers: RACE;
- off-by-one: NOT_EQUIV;
- numerical bugs: EQUIV, a known blind spot of real-number semantics.

**Bug detection, configurations A/B/C** (measured before D2, so "A" here is today's A0, fixed tests only; the D2
re-run on tirx is under "Compiler-checks gate" below):

| bug set | A: fixed tests | B: + VOLTA | C: + VOLTA + change-aware |
|---|---|---|---|
| 11 injected CUDA bugs (softmax, LayerNorm, GEMM) | 4/11 | 5/11 | **11/11** |
| 4 injected tirx-lite bugs (softmax) | 1/4 | 1/4 (VOLTA n/a for tirx) | **4/4** |

The two mechanisms complement each other:
- only VOLTA caught `layernorm_bug_nosync`, a race the GPU tests miss;
- only the change-aware tests caught the two numerical bugs, which VOLTA cannot see (real numbers);
- VOLTA caught the tail and guard bugs only at the change-aware shapes.

**Change classifier, mutation benchmark** (`akc mutants`, CPU).
- **How the edits are made.** Fixed mutation operators generate single edits, each with a set of acceptable
  categories:
  - comparison flips, +1 in a subscript, changed constants, `#define` tile sizes;
  - a deleted barrier, changed warp shuffles;
  - changed float arithmetic, max ↔ min, changed initial values;
  - negated conditions, halved shared memory, float → double;
  - two no-op edits: rename and reformat.
- **On which kernels.** Our 3 optimized kernels, 4 GLM-written CUDA softmax kernels and the tirx-lite reference.
  The rules were revised while looking at these, so their "after" numbers are training-set numbers.
- **The honest estimate** is the **held-out** row: 8 GLM-written kernels (4 CUDA, 4 tirx-lite) that were never
  looked at while revising.

| kernels | classifier | right category | no category | no-op edits labeled |
|---|---|---|---|---|
| tuning set | before (2026-11-15 rules) | 132/198 (67%) | 60 | 25/31 |
| tuning set | after | 198/198 | 0 | 0/31 |
| **held out** | before | 117/189 (62%) | 70 | 16/28 |
| **held out** | after | **189/189** | **0** | **0/28** |

The edits are single, local changes. Real agent revisions are often 10–50-line rewrites; the rules then label most
categories, so the tests are broad rather than targeted. Results: `results/eval/mutants_rules_{before,after}_*.json`.

**IR vs AST classifier on the tirx-lite rows** (`akc mutants --classifier ir|ast`, CPU, 2026-11-17; 106 tirx edits:
24 on the reference, 82 on the 4 held-out GLM tirx kernels):

| kernels | classifier | right category | wrong | no category | no-op edits labelled | AST fallback | `ir_blind` |
|---|---|---|---|---|---|---|---|
| tirx reference (tuning) | AST | 21/21 | 0 | 0 | 0/3 | – | – |
| tirx reference (tuning) | IR | 21/21 | 0 | 0 | 0/3 | 2 | 0 |
| 4 GLM tirx (held out) | AST | 70/70 | 0 | 0 | 0/12 | – | – |
| 4 GLM tirx (held out) | IR | 70/70 | 0 | 0 | 0/12 | 8 | 0 |

- Agreement: every edit has the same verdict in both (91 hit/hit, 15 clean/clean).
- The category sets differ on 36 of 106 edits, the first category on 16. IR adds categories more often than it
  drops them, mostly numerical from the reduction rule. Examples:
  - a guard or loop bound inside a max / sum loop is indexing (or tiling) + numerical;
  - a shuffle offset `16 → 8` in a warp max is sync + numerical (AST: indexing);
  - BLOCK 128 → 64 is tiling + indexing + memory + numerical (AST: tiling).
- Cost of the broader labels: 1409 generated tests vs 1027 with the AST (+37%).
- The 10 AST fallbacks are all `float32 → float16` edits that TIRx refuses to trace.
- Analysis time: 1.1–1.4 s per edit when one child traces all edits of a base. A single revision costs about
  7.7 s (softmax reference vs a bug, 5 shapes), mostly the tvm import; the AST path takes 0.01 s.
- Not fully held out:
  - before one fix the IR scored 69/70 held out;
  - the fix pairs a deleted and an added statement from the same source line. It came from that miss:
    `red_max[w]` → `[w + 1]` in an unrolled loop showed up as "first iteration deleted, extra iteration added".
- Results: `results/eval/mutants_rules_ir_{bases,heldout}.json`.

**Detection on the mutants, A vs C** (`akc mutants --detect --per-op 2`, A100, jobs 24051467 / 24052255; C here is
fixed + change-aware tests, without VOLTA).
- Only base kernels that pass on their own are used. 3 of the 8 held-out GLM drafts already fail the fixed tests,
  so 5 remain.
- Edits that cannot change any output (an array declared 1 larger, float → double) are not counted.
- No-op edits (rename, reformat) raised no false alarm in A or C.

| kernels | edits | A: fixed tests | C: + change-aware tests | false alarms on no-op edits |
|---|---|---|---|---|
| tuning set (8 kernels) | 136 | 83 (61%) | **105 (77%)** | 0/23 |
| held out (5 correct GLM kernels) | 67 | 45 (67%) | **53 (79%)** | 0/11 |

What C's tests still miss (45 edits), checked with VOLTA where possible (job 24052791, CUDA only):
- **Races and out-of-bounds shared memory, 14 CUDA edits.** VOLTA catches all 14: deleted barriers,
  `BLOCK / 2` → `BLOCK / 1`, halved shared arrays, and `if (lane == 0)` negated. In the last case every lane
  writes the same value to the same address: VOLTA reports a race, but the outputs do not change. With VOLTA,
  configuration C catches 119 of the 136 tuning-set edits.
- **Equal in exact math, NaN in float: 8 CUDA edits that neither tests nor VOLTA catch (open item).**
  - In each, the warp max reduction starts its shuffle loop at offset 8 instead of 16, so the row max covers only
    16 of the 32 lanes.
  - Softmax gives the same result for any subtracted value, so over the reals the output is unchanged. VOLTA
    proves 6 of these equivalent. 1 is in `glm_A_r1`, where VOLTA cannot decide. On 1 in `glm_B1_r1`, VOLTA
    reports the base kernel's own race.
  - In float32, if the true max sits in an unseen lane and is more than ~88 above the partial max, exp overflows.
    A CPU model of the edit returns NaN for a row of zeros with a single 100.
  - C classifies these edits as indexing. It gives them normal-valued boundary shapes plus one 2-row `large` floor
    test, and none of those rows has such an outlier.
  - This is the real-number blind spot that the tests are meant to cover. The fix is the `outlier` input family
    (§2 Task 2): one large value per row, at every position modulo the tile across rows. The IR classifier also
    labels a changed shuffle in a reduction as numerical, so these edits now get it. Not re-run on the GPU yet.
- **No change in this kernel: 9 CUDA edits.**
  - VOLTA proves 5 equivalent: xor → down shuffles where only lane 0's value is used, and an extra GEMM K-tile
    whose guarded loads give zeros.
  - 4 more are equivalent by reading the code:
    - 3 are in `glm_A_r1`, where VOLTA cannot decide because its `-FLT_MAX` overflows VOLTA's rationals;
    - 1 is a second-stage reduce over 4 values that starts at offset 8, in `glm_B1_r1`.
- **tirx-lite, 14 edits (VOLTA does not run on tirx yet).**
  - 5 are equivalent: block size changes in kernels that derive everything from the block size.
  - 4 are the same-value races above; 1 is a real race (`tid < s` → `tid <= s` in a tree reduction).
  - 2 read one element past the row inside a max pass. Outputs change only when the next row starts with a value
    far above this row's max: every exp then underflows and the result is 0/0. The `outlier` family targets this
    case: adjacent rows are scaled 1 and 1e3. Not re-run yet.
  - 2 change outputs by less than softmax's tolerance (rtol 1e-4):
    - `[r, col]` → `[r, col + 1]` in a sum over 12 285 elements;
    - a float16 sum accumulator.

The GPU runs also found three problems in the benchmark and the generator; all are fixed:
- **A kernel that never stops.** `off > 0` → `off >= 0` hung the runner for 900 s, twice. Now there is a 180 s
  limit, and C is not re-run when A already failed.
- **Broken base kernels counted.** 3 held-out GLM kernels are wrong by themselves, so all their mutants counted
  as "caught". They are now excluded.
- **A path no test reached.** GLM's large-row code path (N > 8192 in CUDA, n > 12284 in tirx) was never run. This
  led to the size-regime tests (§2): 2 edits there are now caught.

  That fix came from looking at held-out misses, so the held-out detection numbers are no longer fully held out.

Results: `results/eval/mutants_detect_{bases,heldout}.json` (the first run, before the fixes:
`..._v1.json`) and `results/eval/mutants_volta_misses.json`.

**Compiler-checks gate (D2) on tirx-lite, A100** (`scripts/experiments/gate_tirx.py`, job 24061793;
`results/eval/gate_tirx.json`; caught = a report of the tool on our kernel). The first time compute-sanitizer runs on
TIRx-generated code:

| kernels | memcheck | racecheck | initcheck | synccheck | gate (any) |
|---|---|---|---|---|---|
| 5 tirx refs / opts (false alarms) | 0/5 | 0/5 | 0/5 | 0/5 | **0/5** |
| 4 GLM tirx kernels (false alarms) | 0/4 | 0/4 | 0/4 | 0/4 | **0/4** |
| 11 injected tirx bugs | 3/11 | 3/11 | 0/11 | 0/11 | **6/11** |
| 8 tirx mutants VOLTA calls RACE / OUT_OF_BOUNDS | 2/8 | 3/8 | 2/8 | 0/8 | **4/8** |
| 6 tirx mutants VOLTA calls EQUIV | 1/6 | 0/6 | 0/6 | 0/6 | 1/6 |

- **Injected bugs:** racecheck catches the 3 nosync bugs, including `layernorm_txl_bug_nosync`, which every GPU test
  misses; memcheck catches the 2 off-by-ones and the gemm guard. The 5 value bugs (3 tails, nomax, onepass) need
  the tests or VOLTA.
- **Mutants:** the 4 caught are `tid < s` → `tid <= s` (race + uninitialized shared read), BLOCK 128 → 64
  (uninitialized shared read) and 2 GLM `<` → `<=` edits (out-of-bounds read + race). The 4 missed are all
  `if (lane == 0)` negations: every lane writes the same value, so the outputs never change (same as in the CUDA
  trial).
- **Beyond VOLTA:** `glm_txl_C_r2` with `[r, col]` → `[r, col + 1]` reads one element past `x` on GLM's large-row
  path (N = 12285, a gate shape from the size-regime rule). VOLTA said EQUIV: it checks CTA 0 at N ≤ 584 only.
- **Locations:** every report points at a line in the kernel body of the generated CUDA (both accesses of a race),
  after mapping TIRx's inlined helpers back to their call sites.
- **Cost per revision** (median of 34): **10.9 s** (max 12.7 s). Static checks 7.8 s (one child: tvm import 1.7 s,
  two sm_80 compiles 2.8 s), nvcc 0.7 s, the 4 tools in parallel 2.1 s (one after another: 4.5–4.7 s, same
  verdicts). Each tool takes 1.5–2.7 s.
- **CPU-only path** (NumSim, smoke job 24061664): gemm nosync → race at the Python lines of both accesses, gemm guard
  → out_of_bounds, gemm opt clean; 12 s per kernel once NumSim's engine is built (the first synccheck built it:
  44 s).

**Configurations A0 / A / B / C on the 11 injected tirx bugs** (`akc eval-bugs --lang tirx --device gpu`, job
24061793, `results/eval/bugset_tirx_gpu_20261117-045938.json`):

| | A0: fixed tests | A: + gate | B: + VOLTA | C: + change-aware tests |
|---|---|---|---|---|
| detected | 3/11 | 6/11 | 6/11 | **11/11** |
| validation seconds per bug | 8.6 | 18.6 | 21.4 | 41.8 |

- The gate adds layernorm nosync, gemm guard and softmax offbyone to the fixed tests.
- B adds nothing over A on this set: VOLTA at the default shape finds only bugs the gate already found.
- Caveat: for tirx, VOLTA in C runs at B's default shape (`changeaware.plan()` suggests VOLTA shapes only for
  CUDA); see Open items.

**GLM-5.3 as the kernel writer (pilot).** It wrote correct softmax kernels in CUDA (4/4 runs) and in tirx-lite
(4/4) within 3 rounds. It also produced two real, non-injected bugs:
- a shared-memory race: caught only by VOLTA; every GPU test passed;
- a dropped term in an unrolled loop that runs only for N > 384: caught only by the GPU tests at large N. After this
  we added a "deep" VOLTA shape.

**Speed.** rows=4096, N=1024; GEMM 1024³; median of 50 launches, L2 flushed.

| kernel | time | × PyTorch | % of peak |
|---|---|---|---|
| softmax_opt (CUDA) | 0.038 ms | 0.96× | 57% bandwidth |
| softmax_txl_ref (tirx-lite) | 0.033 ms | 1.08× | 65% bandwidth |
| best GLM softmax (CUDA) | 0.034 ms | 1.07× | 64% bandwidth |
| layernorm_opt | 0.038 ms | 1.12× | 57% bandwidth |
| gemm_opt | 0.61 ms | 0.28× | 18% FP32 (PyTorch: 64%) |

A device copy of the same bytes reaches about 66–68% of the datasheet bandwidth. So softmax and LayerNorm are near
the practical limit, and GEMM is where an optimizing agent can gain the most.

**Problems the GPU runs found, all fixed and re-verified:**
- **A false alarm.** The change-aware LayerNorm "offset" test (mean 1e4) rejected the correct `layernorm_opt`:
  rounding the row mean to float32 is unavoidable. Fixed with a condition-aware tolerance; real numerical bugs are
  still caught.
- **A crash.** `eval-bugs --lang tirx` crashed. Fixed.
- **Duplicate feedback lines.** Fixed.

### Where results are
| results | location |
|---|---|
| Slurm logs | `logs/slurm/` |
| bug-set results | `results/eval/` |
| speed | `results/perf/` |
| VOLTA runs | `results/volta/` |
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
