**Design an Agentic AI Kernel Compiler: Progress Report**
Hongyi Chen (hongyic5@illinois.edu) · Minghao Fang (mf46@illinois.edu) · Xiaocheng Zhang (xz105@illinois.edu)

**1 Progress Summary**

We have a working validation pipeline for agent-written GPU kernels, tested on one A100. Three validation blocks are
implemented and connected to an agent loop: (1) compiler checks and fixed tests, the CAKE-style baseline; (2) VOLTA
equivalence checking; and (3) change-aware tests. We tested them on 11 injected bugs and on 34 hard bugs made by
automatically editing correct kernels. Together, the three blocks catch all 11 injected bugs and 33 of the 34 hard
edits, with no false alarms. The loop runs end to end with a scripted model; real-LLM runs are next.

*Change from the proposal.* CAKE has no public code, so we use tirx-lite in its place. tirx-lite is a small Python
kernel language built on TIRx, from MLC's TIRx-harness. The agent (GLM-5.3 via Lumen) writes tirx-lite, and our
checks work on its IR and on the CUDA it compiles to.

**2 Implementation**

*Task 1: Environment and baseline (done).* We wrote softmax, LayerNorm and GEMM in tirx-lite. Each has a PyTorch
(float64) reference, a tirx-lite reference for VOLTA, and 3–4 injected bugs (11 in total). We also built a
correctness harness and a benchmark harness. A fresh clone sets up with one command and passes all tests. For
Block 1, we got compute-sanitizer working on TIRx-generated code; this needed mapping its reports through TIRx's
inlined helpers back to kernel lines. Block 1 raised no false alarms on 9 correct kernels. Baseline speed vs.
PyTorch is 1.08× for softmax, 1.07× for LayerNorm and 0.29× for GEMM. We confirmed that VOLTA can analyze all three
kernels.

*Task 2: Change-aware test generation (done).* We implemented the IR diff, the change classifier and the
per-category test generator, all with unit tests. On all 164 generated edits (Section 3), every edit got the label
that matches its type, and harmless renames got none. Studying the misses in our GPU runs led to two additions:
- tests on both sides of each size threshold in a kernel (one GLM kernel's large-row code path had never been run
  by any test);
- outlier inputs.

The generator also picks the sizes VOLTA checks in configuration C. An optional LLM second opinion for the labels
is built but has not yet been tried with real calls.

*Task 3: VOLTA integration (done).* We made VOLTA read TIRx-generated code (five rewrites, then PTX) and turned its
verdicts into hints for the agent. All 112 VOLTA runs on the three kernels' references, optimized versions and bugs
gave a verdict. VOLTA cannot yet decide GEMM edits that change the tile size.

*Task 4: Agent feedback and repair loop (implemented).* In our loop, the first failing block stops validation, and
its report, with a hint, becomes the next prompt. A passing candidate is timed. The loop runs for up to 10
iterations or until a coin budget is used, and it logs every iteration. We tested it with a fake model (10 unit
tests), and on the A100 with a scripted model that starts from an injected bug. In all three runs (one per kernel),
the bug was caught by the expected block and the loop ended after one revision. An earlier pilot showed that hints
matter: a raw VOLTA error made GLM-5.3 break a correct kernel, while a one-line hint got it fixed in one round. We
have not yet run this loop with the real LLM.

*Task 5: Evaluation (in progress).* We built the detection workload and ran configurations A, B and C on it
(Section 3). This run exposed a bug in our test harness: misaligned-pointer tests were counted as catches. We fixed
it and re-ran the affected edits. Repair success and final kernel performance need real-LLM runs and are not
measured yet.

All components have unit tests (244, all passing).

**3 Testing and Preliminary Results**

We first measure how well each configuration finds wrong kernels. Wrong agent kernels are rare (GLM wrote correct
softmax in 4 of 4 pilot runs), and their true answer is unknown. So we use one fixed workload whose answers we know,
the same for every configuration:
- 11 injected bugs. Each kernel has a tail bug (a loop skips the last N mod tile elements, so the bug only shows at
  sizes that are not a multiple of the tile), an off-by-one (`<=` instead of `<`) and a missing barrier. Two more
  are numerical: softmax without subtracting the row max, and LayerNorm variance computed as E[x²] − mean².
- 34 hard edits. We take 9 correct kernels (our references and optimized versions, plus 4 softmax kernels written
  by GLM) and automatically change one place in each copy, the way an agent might slip. This gives 164 edits. We
  drop the 119 that plain fixed tests already catch (36 do not even compile), and the 11 that cannot change any
  output (e.g. a tile-size change in a kernel that derives everything from the tile size). The 34 left are:
  - 18 boundary slips (`<` → `<=`, a halved block size, an index +1) that read out of bounds or uninitialized
    memory while the output still looks right;
  - 9 races: 2 deleted barriers, and 7 negated `if (lane == 0)` guards, after which every thread writes the same
    value;
  - 2 halved shared arrays, which overrun shared memory;
  - 5 numerical edits, e.g. max → min, or a max initialized to 0 instead of −FLT_MAX.
- 27 renames, which change nothing, so any flag is a false alarm.

Configurations follow the proposal:
- **A** = Block 1: the proposal's fixed tests, plus the compiler checks CAKE already runs;
- **B** = A + VOLTA at one default size;
- **C** = B + change-aware tests, with VOLTA at the sizes the change-aware tests pick.

*Table 1. Results on the workload (one A100, 2 h 10 min).*

| | A | B | C |
|---|---|---|---|
| Injected bugs caught (/11) | 6 | 6 | 11 |
| Hard edits caught (/34) | 19 | 29 | 33 |
| False alarms (/27) | 0 | 0 | 0 |
| VOLTA coverage (decided / run, all 164 edits) | – | 86% | 86% |
| Time per candidate | 19 s | 22 s | 50 s |

- **A.** Fixed tests alone catch only 3 of the 11 bugs. The compiler checks add 3 bugs and 19 hard edits:
  out-of-bounds and uninitialized reads, and races, whose outputs still look right.
- **B.** VOLTA adds 10 hard edits. No test in any configuration catches 9 of them; most are races in which all
  threads write the same value. At its single default size, VOLTA misses the tail bugs.
- **C.** The change-aware tests add 5 bugs (the 3 tail bugs and the 2 numerical bugs) and 4 hard edits, all
  float-only failures. On the injected bugs, the change-aware tests catch all 11 and VOLTA at their sizes catches 9.
  VOLTA misses the two numerical bugs because it reasons over real numbers, so the two mechanisms are
  complementary.
- C misses one hard edit: an index +1 that reads out of bounds only at N = 1, with the output still within
  tolerance.

**4 Remaining Work**

Run the loop with GLM-5.3 under configuration C to repair the 11 injected bugs and to optimize the three kernels,
measuring repair success and final speed. For each agent candidate, also record what A and B would have said. Then
compare final kernel speed across A, B and C.
