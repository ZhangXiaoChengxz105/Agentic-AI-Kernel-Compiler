# CS 521 Project: Agentic AI Kernel Compiler (CAKE + VOLTA + change-aware testing)

## 0. HARD RULES: scope and side effects (read first, always follow)

**Workspace root** = the directory containing this file (also `$CS521` after `source env.sh`).

1. **Write only inside the workspace root.** Create, modify, or delete files ONLY under this directory.
   Do NOT touch anything outside it: no edits to `~/.bashrc`, `~/.profile`, `~/.gitconfig`, `~/.ssh`,
   `~/.config`, other folders in home, shared `/projects` dirs, or system paths.
2. **Install only inside the workspace root.**
   - Python: use an env created inside this folder (`python -m venv ./.venv` or `conda create -p ./envs/<name>`).
     Never `pip install --user`, never install into conda `base`, never run `conda init`.
   - npm: always `source env.sh` first (prefix and cache already point here).
   - Caches (pip, conda pkgs, Triton, Torch, CUDA JIT) must go inside this folder; see env.sh.
3. **No external effects without explicit approval.** Before doing any of the following, STOP and ask,
   stating *what*, *why*, and *cost/impact*:
   - Submitting Slurm jobs (`srun`, `sbatch`, `salloc`): this consumes a shared GPU allocation.
   - Calling Lumen or any LLM/remote API: this consumes coins/quota.
   - `git push`, opening PRs/issues, sending email or messages, any network *write*.
   - Deleting or overwriting data that you did not create in this session.
   - Running heavy compute on the login node (login nodes are shared and cgroup-limited).
4. Read-only actions are fine (reading docs, `module avail`, `nvidia-smi` on a granted node,
   downloading/cloning source *into* this folder).
5. **If something unavoidably touches outside this folder, say so explicitly before doing it:**
   "This will write to <path> because <reason>. Proceed?" Then wait for approval.
6. When unsure about CAKE or VOLTA behavior, read their source/docs; do not guess.
   Before multi-file changes, propose a short plan first.

## 1. Project summary (from our proposal)

**Problem.** AI agents can generate and optimize GPU kernels, but an optimized kernel may silently
break semantic equivalence with its reference or introduce data races/deadlocks. Passing a fixed
test set does not prove correctness; typical failures are wrong boundary handling after tiling
changes and synchronization errors after instruction reordering.

**Baseline system: CAKE.** Compiler–agent co-design: the agent writes kernels in CAKE IR; the
compiler checks the IR, lowers it to GPU code, and returns diagnostics so the agent can refine.

**Our extension.** Add two complementary validation mechanisms to CAKE's loop:
1. **Change-aware test generation.** After each agent revision, diff the CAKE IR, classify the change
   (tiling / memory access / synchronization / numerical), and generate targeted inputs:
   tiling/indexing → boundary and non-divisible shapes; sync → varied sizes and pipeline stages;
   numerical → edge-case values.
2. **VOLTA verification.** For candidates that pass CAKE's checks and fall within VOLTA's supported
   scope, check semantic equivalence. VOLTA uses a real-number model.
Counterexamples and test failures are turned into textual diagnostics for the agent, which revises
and repeats until it passes or hits the iteration cap.

**Evaluation.** Compare 3 configurations: (A) fixed tests; (B) fixed tests + VOLTA;
(C) fixed tests + VOLTA + change-aware tests. Metrics: error detection, verification coverage,
validation overhead, final kernel performance, repair success.

**Concrete setup (from our response to instructor feedback).**
- GPU: single NVIDIA A100 (NCSA Delta) for compilation, testing, benchmarking.
- LLM agent: open-weight models via Lumen (lumen.ncsa.illinois.edu, OpenAI-compatible API).
- Kernels: 2–3 of tiled GEMM, row-wise softmax, LayerNorm, each with a PyTorch reference.
- Agent loop: max 10 iterations per optimization run.
- Bug injection: manually inject typical bugs (off-by-one boundary, missing barrier, etc.)
  into correct kernels to measure detection, since the agent may rarely produce wrong kernels.

## 2. Key understanding

- In essence this is a counterexample-guided repair loop: the agent proposes, validators search
  for counterexamples, and feedback drives revision.
- **VOLTA and tests are complementary.** VOLTA gives proofs, but only within its scope and under
  real-number semantics, so it is blind to floating-point issues and unsupported kernels.
  Change-aware tests cover exactly that gap, and are cheaper and more targeted than a fixed test suite.
- **Biggest risks:**
  1. VOLTA may not support all chosen kernels (verify first).
  2. CAKE's target GPU architecture and CUDA version must match the hardware.
- "Verification coverage" must be defined precisely (e.g., fraction of candidates VOLTA can decide).

## 3. Task breakdown (Xiaocheng owns Tasks 1–2 first)

### Task 1: Environment and baseline (Week 1)
- 1.1 Fetch CAKE and VOLTA source into `third_party/`. Record target GPU arch, CUDA/Python versions,
      and dependencies in `notes/env.md`.
- 1.2 **VOLTA feasibility check (do early):** can VOLTA analyze GEMM / softmax / LayerNorm as
      expressed in CAKE? Record the result in `notes/volta_scope.md`.
- 1.3 Create an isolated Python env inside this folder and extend env.sh as needed.
- 1.4 PyTorch reference implementations in `ref/` (GEMM, softmax, LayerNorm), including
      non-divisible shapes.
- 1.5 Correctness harness in `harness/` (compare against reference with dtype-aware tolerances).
- 1.6 Benchmark harness (CUDA events, warmup, median of N runs).
- 1.7 Slurm scripts in `scripts/slurm/` (1 GPU, tight `--time`). **Ask before submitting.**
- 1.8 Lumen client wrapper (key from `$LUMEN_API_KEY`). **Ask before any real API call.**

### Task 2: Change-aware test generation (Weeks 2–3)
- 2.1 IR diff module: compare CAKE IR before/after an agent revision and list the changed ops.
- 2.2 Change classifier: tiling / memory access / synchronization / numerical.
- 2.3 Test generator per category:
      tiling/indexing → boundary and non-divisible dims; memory → misaligned/strided cases;
      sync → varied sizes and pipeline stages; numerical → large/small magnitudes,
      softmax overflow, subnormals.
- 2.4 Unit tests for the diff, classifier, and generator.

### Task 3: VOLTA integration (Weeks 2–3, parallel)
- Invoke VOLTA on supported compiled kernels; collect results and counterexamples;
  convert them into structured feedback for the agent.

### Task 4: Agent feedback and repair loop (Week 4)
- Wire tests and VOLTA into the CAKE loop; format diagnostics; cap at 10 iterations; log every iteration.

### Task 5: Evaluation (Weeks 5–5.5)
- Build the bug-injection set (off-by-one, missing barrier, wrong sync order, tiling boundary).
- Run configs A/B/C; report detection, coverage, overhead, performance, repair success.

## 4. Suggested layout
