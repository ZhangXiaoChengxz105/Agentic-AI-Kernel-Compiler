# Progress so far: slide material (2026-11-17)

Three slides. Each has the text for the slide, then "Notes" with details for the speaker or for a backup slide.
All numbers are from `README.md` §2 and §4 and `notes/gpu_results.md`.

---

## Slide 1: The agent loop we need

**Overall input**
- Task: the operator's description, its tirx-lite function signature and a shape contract (the legal shape range,
  e.g. softmax rows 1–4096, N 1–16384, and one benchmark shape for speed).
- Correct answer: a PyTorch reference (computed in float64). VOLTA also gets a simple tirx-lite reference kernel.
- Optional start kernel (e.g. an injected bug, to measure repair).
- Configuration (A / B / C), an iteration cap of 10, and a coin budget.

**Overall output**
- The best correct kernel: the fastest candidate that passed the configuration's checks, with its speed vs PyTorch.
- A log of every iteration: prompt, model reply, each check's report, the feedback text, and why the loop stopped.

**The loop**
```
 task + contract ──► LLM agent (GLM-5.3 via Lumen) ──► candidate kernel (tirx-lite)
        ▲                                                   │
        │                                                   ▼
        │                         ┌──────────── validation (the configuration) ───────────┐
        │                         │ Block 1  compiler checks + fixed tests   (A, B, C)    │
        │                         │ Block 2  change-aware tests              (C only)     │
        │                         │ Block 3  VOLTA                           (B, C)       │
        │                         └───────────────────────────────────────────────────────┘
        │                              fail │                            pass │
        │                                   ▼                                 ▼
        └──────── feedback text ◄── error report + hint          speed measurement (vs PyTorch)
                  (next prompt)                                   stop, or keep optimizing to 10
```
The parent of each candidate is the last candidate that passed (the reference for the first one).

**Each block is its own flow**
- **Block 1: compiler checks + fixed tests** (all configurations; CAKE's existing checks)
  - In: the candidate kernel.
  - Flow: static checks (safety scan → trace with the IR contract check → IR well-formed and memory checks →
    compile for sm_80) → compute-sanitizer on the generated CUDA (memcheck, racecheck, initcheck, synccheck) →
    fixed tests (3 power-of-two sizes, normal data, 3 repeats) against PyTorch.
  - Out: compile error / race / out-of-bounds / uninitialized read / wrong output, with the line and a hint.
- **Block 2: change-aware tests** (C only)
  - In: the candidate and its parent.
  - Flow: diff the two tirx-lite IRs → label each change (tiling, indexing, memory, sync, numerical) →
    generate tests for those labels → run them on the GPU.
  - Out: failing tests (size, input type, got vs expected, the changed line that likely caused it), and the sizes
    VOLTA should check.
- **Block 3: VOLTA** (B and C)
  - In: the candidate, the tirx-lite reference, and sizes (B: one default size; C: the sizes Block 2 picked).
  - Flow: compile both to CUDA → rewrite it so VOLTA can read it → `volta compare` (and `volta analyze` for
    out-of-bounds shared-memory accesses).
  - Out: EQUIVALENT / NOT EQUIVALENT (with a counterexample) / RACE / OUT OF BOUNDS / UNSUPPORTED, plus a hint.

**Configurations = which blocks run:** A = Block 1; B = Block 1 + Block 3; C = Block 1 + Block 2 + Block 3.
Order in C: Block 1 → Block 2 → Block 3. In the loop the first failing block stops the validation, so the agent
gets feedback quickly.

**Status:** the loop is implemented (`akc agent`). It was tested with a fake model (10 unit tests) and on the A100
with a scripted model: an injected bug as the start kernel, the reference as the reply (job 24078880).
- softmax tail bug, config C: caught by the change-aware tests; iteration 1 passed.
- LayerNorm nosync, config B: caught by racecheck; iteration 1 passed.
- GEMM nosync, config A: caught by racecheck; iteration 1 passed.

It has not yet run with the real model; that needs Lumen coins.

### Notes
- Stop rules: the candidate passes (or, in optimize mode, keep improving speed until 10 iterations), 10 iterations,
  the coin cap, or a model error.
- Context is rebuilt every turn instead of growing a chat: task + the previous draft + its feedback + the best
  correct version so far.
- Feedback lists errors first: which check, what failed, on which input, an example element (got / expected), the
  related changed line, and a hint. Then it says what could not be verified, so the model knows what a "pass"
  covers. If the candidate passed, the feedback also gives its speed: time, × PyTorch, and % of the GPU's peak.
- Why hints matter (GLM pilot): a raw VOLTA error string made the model rewrite a correct kernel and break it; the
  same verdict with a one-line hint was fixed in one round.
- Validation time per iteration: about 9 s (A0), 19 s (A), 22 s (B), 50 s (C). Speed measurement: about 6 s.

---

## Slide 2: Building blocks and progress

**Foundation (done)**
- softmax, LayerNorm and GEMM in tirx-lite (our CAKE substitute: CAKE has no public code). Each has a reference
  and 3–4 injected bugs (11 in all); LayerNorm and GEMM also have an optimized version.
- Speed vs PyTorch: softmax 1.08×, LayerNorm (optimized) 1.07×, GEMM (optimized) 0.29×. GEMM leaves the most room
  for the agent.

**Block 1: compiler checks + fixed tests**: done
- In: the candidate. Out: compile errors, races, out-of-bounds and uninitialized reads, wrong outputs.
- First time compute-sanitizer ran on TIRx-generated code. Reports are mapped back to the kernel's lines.
- 0 false alarms on 9 correct kernels.
- Injected bugs: 6/11 (fixed tests alone: 3/11).
- About 11 s for the compiler checks and 9 s for the fixed tests, per revision.

**Block 2: change-aware tests**: done
- In: the candidate + the last passing version. Out: targeted failing tests, and sizes for VOLTA.
- The diff works on the tirx-lite IR and names the exact IR node that changed (e.g. a loop bound `For.end`, or a
  guard `If.condition: < → <=`).
- Labels are right on every generated edit: 21/21, 70/70 and 73/73 across the three sets. No-op edits got no label:
  0/27.
- Tests per label:
  - tiling / indexing: sizes that are not multiples of the tile;
  - sync: large sizes and repeated runs;
  - numerical: extreme values and outliers;
  - every revision also gets a small "floor" set.
- About 20 s per revision (8 s of it is the IR analysis).
- Not done: the optional LLM second opinion for the labels is built but not tried with real calls.

**Block 3: VOLTA**: done
- In: the candidate, the reference and sizes. Out: equivalent or not (with a counterexample), race, out-of-bounds,
  unsupported.
- Runs on the CUDA that TIRx generates, after 5 rewrites (3 exact, 2 modelling choices).
- All three kernels can be analyzed: 69/69 softmax runs decided, 18/18 LayerNorm, 25/25 GEMM.
- Decides all 34 hard edits, and 86% of all generated edits (114/132).
- About 3 s at one size and 9–13 s at the change-aware sizes.
- Limits:
  - real-number model, so it cannot see float problems;
  - checks thread block 0 only;
  - exact sizes only;
  - GEMM tile-size changes are undecided.

**Agent loop**: implemented and tested with a scripted model. Not yet run with the real model.

### Notes
- Change-aware details: the IR is traced at up to 8 sizes in the shape range (benchmark size, N = 1, T−1, T+1,
  2T+1 for tile T, and both sides of every size threshold in the kernel). If tracing fails it falls back to a Python
  AST diff.
- IR vs AST labels agree on all 106 edits checked. The IR gives broader labels, so it generates 37% more tests.
- VOLTA rewrites:
  - exact: `expf` → VOLTA's exp, a counted barrier → `bar.sync 0`, compare-select → `max`;
  - modelling choices: the fast `ex2.approx` is read as exp, and a constant ≥ 2^127 (the identity of a max) is
    read as ∞.
- Known gaps:
  - one real miss: an out-of-bounds read at N = 1 whose output stays within tolerance;
  - the held-out set is partly inspected;
  - all generated edits are single small changes.

---

## Slide 3: Test configurations and how well they catch errors

**Configurations**
- **A0**: fixed tests only (ablation).
- **A**: compiler checks + fixed tests, i.e. the CAKE baseline.
- **B**: A + VOLTA at one default size.
- **C**: A + change-aware tests + VOLTA at the change-aware sizes.

**Why test detection first.** We first measure how well each configuration catches errors, using kernels we know
are wrong. Inside the agent loop this is hard to measure:
- wrong candidates are rare (GLM wrote correct softmax kernels in 4 of 4 pilot runs);
- for an agent's candidate we do not know the true answer.

**Test sets**
- 11 injected bugs:
  - softmax: tail, offbyone, nosync, nomax;
  - LayerNorm: tail, offbyone, nosync, onepass;
  - GEMM: ktail, guard, nosync.
- 34 hard edits: generated single edits of correct kernels that change the kernel's behaviour and that fixed
  tests miss. From 164 generated edits we archived 119 that plain fixed tests already catch (36 do not compile) and
  11 that cannot change any output. Three sets:
  - tuning: the softmax reference, 6 hard edits;
  - held out: 4 GLM-written kernels, 13;
  - fresh: the LayerNorm and GEMM references and optimized versions, 15.
- 27 no-op edits (renames), to count false alarms.

**Detection** (caught / total; A100, 2 h 10 min of GPU time)

| candidates | A0 | A | B | C |
|---|---|---|---|---|
| 11 injected bugs | 3 | 6 | 6 | **11** |
| 34 hard edits | 0 | 19 | 29 | **33** |
| false alarms (27 no-op edits) | 0 | 0 | 0 | 0 |

**What each step adds**
- **Compiler checks (A0 → A):** races, out-of-bounds and uninitialized reads whose outputs still look right
  (3 bugs, 19 edits).
- **VOLTA (A → B):** races where all threads write the same value, shared-memory overruns that do not change the
  output, and one value bug (10 edits).
- **Change-aware tests (B → C):** tail bugs, and float-only bugs that VOLTA's real-number model cannot see (5 bugs,
  4 edits).

**VOLTA and the tests are complementary.** On the injected bugs, the change-aware tests catch 11/11 and VOLTA (at
C's sizes) 9/11; VOLTA misses the two numerical ones.

**C's one miss:** the N = 1 out-of-bounds read, whose output stays within tolerance. The 11 archived edits that
cannot change any output (tile or block size changes, zero-padding values) are flagged by no configuration.

**Cost per candidate:** A0 9 s, A 19 s, B 22 s, C 50 s.

### Notes
- Every candidate runs through the same validation code as the agent loop. For counting, every block runs even
  after a failure. The parent is the correct kernel the edit was made from.
- Verification coverage (VOLTA decided / ran): injected bugs B 11/11, C 10/11 (softmax tail at N = 100 is
  unsupported); hard edits 34/34; all 164 generated edits 86%.
- Example, `softmax_txl_bug_tail` (the loops skip the last N mod 128 elements):
  - fixed tests pass, because they use multiples of 128;
  - VOLTA at the default size N = 256 says "equivalent", so B misses it;
  - the change-aware tests (sizes 127, 129, 1000, …) and VOLTA at N = 584 / 200 catch it.
- A test-harness bug found during the run, and fixed: the "misaligned pointer" tests always fail on tirx-lite,
  because TIRx refuses misaligned tensors even for correct kernels. After the fix, 7 edits went from caught to
  missed; the table is after the fix.
- Next:
  - run the agent loop with the real model, in config C: repair the 11 injected bugs, and optimize the 3 kernels;
  - per candidate, also log what A0 / A / B would have said (detection on the agent's own kernels);
  - compare the final kernel speed under A / B / C. A / B's final kernels are checked with C, the strongest check
    we have.
