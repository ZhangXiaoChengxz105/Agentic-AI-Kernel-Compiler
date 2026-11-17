# Design: from hand-written rules to a general change-aware tester (design only, 2026-11-17)

Status: a proposal. Nothing in this note is implemented.

Inputs:
- the mutation runs: `notes/build_log.md` §8c–8e and `results/eval/mutants_detect_*.json`;
- the checker trial of 2026-11-17: the last section of `notes/gpu_results.md`, and
  `results/eval/sanitizer_trial.json`;
- the tool survey: `notes/related_tools.md`.

Every claim is marked:
- **[measured]**: we ran it;
- **[read]**: we read the code or the docs;
- **[assumed]**: not checked yet.

## 1. Where we are

**What the tester does today [read].**
- The diff of a revision is classified into tiling / indexing / memory / sync / numerical.
- Each category has a fixed recipe:
  - tiling / indexing: boundary sizes around the kernel's own tiles;
  - memory: odd sizes and misaligned pointers;
  - sync: large sizes with repeats;
  - numerical: the manifest's list of value distributions.
- A "floor" set runs on every revision:
  - a few boundary sizes;
  - one extreme distribution;
  - both sides of every size threshold the kernel compares against;
  - since the uncommitted change, `spike` rows.

**Detection [measured]** (C = fixed + change-aware tests, without VOLTA; edits that cannot change outputs are not
counted).

| set | edits | A (v2 / v3) | C before `spike` (v2) | C with `spike` (v3) |
|---|---|---|---|---|
| tuning (8 kernels) | 136 | 83 / 84 | 105 | 112 |
| held out (5 correct GLM kernels) | 67 | 45 / 45 | 53 | 56 |

**What C missed in v2 [measured + read]:**
- Races and out-of-bounds shared-memory accesses that do not change values.
  - 14 CUDA edits. Before the trial only VOLTA saw them; compute-sanitizer now catches 6 (see P3).
  - 5 tirx edits: 4 same-value writes and 1 real race. There is no VOLTA for tirx yet.
- Bugs that exist only in floating point: 8 shift-only CUDA edits and 2 tirx read-past-the-row edits. Since v3,
  `spike` catches all 10.
- Changes below the tolerance: 2 tirx edits, one of them a float16 accumulator.
- Harmless or equivalent edits: 9 CUDA, 5 tirx.

**Two rules were added after looking at misses:**
- size-regime thresholds (§8c);
- `spike` (§8e).

Both raised the held-out numbers. So **the held-out set is no longer fully held out**, and we cannot tell how much
of the gain carries over to new kernels.

## 2. Principles

### P1. Two oracles in every test: the reference and the parent
- **Reference (kept).** Compare with the float64 golden:
  - tolerance (condition-aware, as now);
  - non-finite outputs;
  - unwritten outputs;
  - canaries;
  - nondeterminism across repeats.
- **Parent revision (new).** Run the last accepted revision on the same inputs, in the same process.
  - A test that the parent passes and the candidate fails is a *regression of this diff*. The feedback can then
    name the edit.
  - A test that both fail is an *old* bug. An example is glm_B1_r1's race, which shows up under every mutant of
    that base. Today such failures are attributed to the new edit.
  - When the agent only meant to change speed, outputs that change bitwise on a deterministic input are reported as
    *info*, not as errors. This is how Shadow and Meta's JiT catching tests judge a change (see related_tools.md).
- **Admissible inputs (new, generic).**
  - Rule: an input counts only if a trusted float32 implementation passes on it. We have one per op:
    `OpSpec.torch_impl`, which is PyTorch float32.
  - This rule replaces per-op guesses such as "200 is safe for softmax but would overflow GEMM". Any value is
    allowed if a correct float32 kernel can handle it.
  - [assumed] PyTorch float32 is close enough to "any correct float32 kernel". The LayerNorm offset false alarm
    showed that tolerances must account for conditioning. They already do, through `OpSpec.magnitude`.

### P2. Search over generic input families instead of fixed values
The families are defined only from the shape keys, the reduced axis, and the sizes the kernel itself declares
(`extract_tiles`). None of them knows the operator.

| family | parameters | what it reaches |
|---|---|---|
| F1 dense ranges | scale on a ladder 2^k (tiny to large), constant rows (ties), mixed scales in one row | overflow/underflow of a whole row, cancellation, ties |
| F2 sparse extremes | 1–2 outliers of magnitude ±m (m on a ladder 2^k) on a small background, at position p | reductions that skip elements, max/min/argmax paths, exp overflow |
| F3 neighbour contrast | neighbouring rows (or K-tiles in GEMM) at very different offsets or scales | reads past a row end, wrong row stride, a dropped or duplicated tile |
| F4 boundary sizes | 1, b−1, b+1, 2b±1 for every declared size b; both sides of every size threshold; one prime | tails, guards, the second code path (already generic today) |
| F5 special values | ±0, ±inf, NaN, subnormals, FLT_MAX, only where admissible (P1) | NaN propagation, inf handling |

**Positions in F2 (the general form of spike's column choice).**
- The position classes are p mod b for every b in B:
  - the tile and block sizes found in the kernel;
  - the warp width 32 (a CUDA hardware constant);
  - the vector width 4;
  - plus the first and last element of the axis.
- One outlier goes in each row, at residue (row mod lcm(B)), plus a random multiple of lcm(B). So lcm(B) rows
  (128 for our kernels) cover every residue, and different warps and tiles get the outlier.
- Unlike spike, this does not assume that thread i reads element i.
  - Example: a kernel where each thread reads 8 consecutive elements puts element i in lane (i / 8) mod 32.
  - For such a kernel, residues mod 128 alone reach only lanes 0–15. The random multiple of lcm(B) moves the
    outlier into the other half.
  - So this coverage holds with high probability, not by construction. [assumed]

**Search procedure for each family, within a budget.**
1. Coverage pass (deterministic): every position class times a coarse magnitude ladder, all packed into the rows of
   a few tensors.
2. Refinement: a few rounds of local search around the worst input, adjusting magnitude, position and scale.
   - The score is the largest error divided by the tolerance, plus non-finite counts, plus divergence from the
     parent.
   - Xscope uses Bayesian optimization for this; a simple hill climb is enough to start.
3. Shrinking: reduce a failing input to the fewest rows, the smallest size still past the failing boundary, and a
   single outlier, before it becomes feedback. Hypothesis does this.

**"Every element must matter."** This is spike's real principle, made explicit as a coverage measure: the share of
(position class × value class) cells that some test exercised. It is reported per revision, so that a gap shows
up as a number and not as a miss found later.

### P3. Operator-agnostic runtime checkers, scoped by the trial
Trial: job 24059082, 41 kernels, small shapes, N(0,1) inputs plus `spike` for the shift-only edits. All
numbers in P3 are **[measured]**.

| group | memcheck | racecheck | synccheck | initcheck (shared + global) | any |
|---|---|---|---|---|---|
| 7 correct bases (false alarms) | 0 | 0 | 0 | 0 | **0/7** |
| 11 injected bugs | 3 | 3 | 0 | 0 | 6/11 |
| 14 VOLTA-only race/OOB mutants | 0 | 4 | 0 | 2 | 6/14 |
| 8 shift-only mutants | 0 | 0 attributable | 0 | 0 | 0/8 |

**racecheck, memcheck, and initcheck on shared memory: run them on every revision that passes the value tests.**
- What they catch:
  - every deleted barrier (5);
  - the real write collisions (2 halved `smax` arrays);
  - glm_B1_r1's race;
  - the global out-of-bounds reads and writes (3);
  - shared reads past an array (2, initcheck).
- What they miss:
  - same-value writes. These are harmless: 6 `if (lane == 0)` negations;
  - 2 out-of-bounds writes that stay inside the block's shared window. Only VOLTA sees those;
  - every value-only bug.
- They need the code path to run, not particular values. So the change-aware part's job for them is to pick
  shapes that execute the changed code: tails, and both sides of each threshold.
- They can give the race check to kernels that VOLTA cannot handle: `UNSUPPORTED` kernels, and tirx. TIRx compiles
  to CUDA, so compute-sanitizer should apply. [assumed: not tried on tirx]

**synccheck: optional, it is cheap.** It caught nothing here, because no kernel in the set has a divergent barrier.
An agent could write one, e.g. a `__syncthreads()` inside `if (tid < s)`. [assumed]

**GPU-FPX: a localizer, not a detector.**
- It reports an exception only when the input triggers it. With N(0,1) inputs, the shift-only edits raise no NaN.
- INF alone is noise: correct online-softmax kernels produce −INF on purpose (3 of 5 GLM bases).
- Two signals separated mutants from bases on this set, with no false alarms:
  - NaN;
  - an exception kind the parent does not raise on the same input (P1 applied to the checker).
- It gives a source line, but for none of the GLM-kernel NaNs.
- Its dependency is fragile. It is a patch set over NVBit 1.5.5 and needed a port to NVBit 1.8. NVBit 1.8 claims
  drivers ≤ 575, yet worked on 595.
- So keep it optional. Run it only on inputs that already fail, or when the diff touches math, to name the line.

**Consequence: checkers do not replace input search.** No checker reaches the shift-only edits or the numerical
injected bugs without the right input. P2 is still needed, and the checkers make P2's job smaller: P2 no longer has
to expose races through values.

### P4. VOLTA for equivalence
- VOLTA is the only component that proves "equal over the reals". It also finds races and shared-array overflows
  that the runtime checkers miss.
- Its blind spots are:
  - floating point (the shift-only edits are EQUIV);
  - kernels outside its scope (`-FLT_MAX` in glm_A_r1 is UNSUPPORTED);
  - tirx (not wired up yet).
- P2 and P3 cover these blind spots; VOLTA is not replaced.

### P5. The change-aware part decides where to look, not which values
The diff and the classifier are used only to choose:
- **shapes:** the declared sizes and thresholds that appear in or near the changed lines come first, and get both
  sides of each boundary;
- **code paths:** if the changed lines are inside a branch guarded by a size comparison, sizes on that side of the
  threshold are required;
- **position classes:** the block or tile sizes used in the changed index expressions come first in F2;
- **budget:**
  - more search rounds when a reduction, a math call or float arithmetic changed;
  - fewer when only a tile constant changed;
  - never zero, because the floor coverage pass always runs;
- **checkers:** which shapes the checkers run at. The checkers themselves are cheap enough (about 8 s per tool,
  in parallel) to run racecheck, memcheck and shared-memory initcheck on every revision that passes the value
  tests. GPU-FPX runs only when math changed or a value test failed.

The classifier is right on 189/189 held-out single edits [measured], but real agent revisions are 10–50-line
rewrites. A wrong label should therefore only lower the priority of a family, never remove it.

## 3. Pipeline per revision (sketch)
1. Static step (CPU, under 1 s; kept):
   - diff and classify;
   - `extract_tiles` and `extract_thresholds`;
   - mark the changed lines.
2. Plan:
   - shapes from F4;
   - position classes from B;
   - a budget per family from P5;
   - which checkers to run.
3. Run, in one child process per candidate (kept):
   - candidate, parent and golden on the coverage pass;
   - then the refinement rounds;
   - each input is first checked for admissibility with `torch_impl`.
4. Checkers: compute-sanitizer tools on 2–3 small shapes that cover each code path (both sides of each threshold).
5. VOLTA on the planned shapes (kept).
6. Feedback for each failing check:
   - the shrunk input;
   - parent vs candidate (regression or old bug);
   - the checker's source line.

## 4. What to keep and what to replace

**Keep:**
- `diff.py` and `classify.py`, now as a routing signal rather than a test recipe;
- `extract_tiles` and `extract_thresholds` (the regime fix is already generic: thresholds are read from the
  kernel);
- the checks in `runners/check.py` (sentinel, canaries, non-finite, condition-aware tolerance) and the repeat-based
  nondeterminism check;
- the isolated child-process runner;
- VOLTA and `volta_shapes`;
- the mutation benchmark (`akc mutants --detect`) and `volta_on_misses.py`;
- the trial's launcher in `scripts/experiments/sanitizer_trial.py` (exact-size allocations, no PyTorch GPU kernels)
  and its report parser, as the start of a checker validator.

**Replace:**
- the per-category value recipes and the per-op `numerical_dists` lists in the manifests. These are op knowledge
  written by hand: softmax gets large/neg_large/mixed/constant/spike, LayerNorm gets offset/constant/tiny/large;
- fixed sizes in the recipes (1000, 4096, 8192, 256 rows) that are not derived from the kernel;
- the single "stress" distribution in the floor;
- the `spike` distribution itself, which becomes F2 + F3.

**The spike change, part by part:**
- General, keep the idea:
  - one sparse extreme per row on a normal background (F2);
  - the position moves from row to row so that every lane is hit (F2 position classes);
  - "every element must matter" (the coverage measure);
  - one row that starts far above its neighbour, which is a special case of neighbour contrast (F3).
- Overfit, drop:
  - the value 200. It was chosen to exceed ln(FLT_MAX) ≈ 88 for softmax's exp after reading the misses; a
    magnitude ladder plus the admissibility rule replaces it;
  - the value 400 in row 1, aimed at the tirx max pass that reads past the row end. F3 replaces it;
  - column 32k + (row mod 32). It assumes element i is read by lane i mod 32 (thread i mod 128); position classes
    from the kernel's own sizes replace it;
  - 64 rows. lcm(B) rows replace it;
  - row ops only. GEMM gets no spike at all; F2 and F3 apply to any reduced axis, including GEMM's K.

## 5. Cost per revision
**Today [measured].** All figures are medians per candidate on an A100.

| component | CUDA | tirx |
|---|---|---|
| fixed tests (A) | 3.2 s | 8.4–8.9 s |
| fixed + change-aware tests (C) | 6.4 s | 25–26 s |

- VOLTA: 0.8 s median, 15 s maximum (on the 31 misses).
- Checker trial: about 8 s per tool run (7.5–8.1 s medians, at most 14.6 s). Most of that is process start:
  Python, torch import, CUDA context. A run with no tool took 10.3 s on a cold cache.

**Proposed, CUDA [assumed estimates, to be measured after implementation].**
- Static step: under 1 s.
- Coverage pass plus refinement:
  - parent and candidate run in the same process;
  - inputs are packed as rows of a few tensors (lcm(B) rows × a few shapes × about 5 magnitude levels);
  - about 50–100 small launches, CPU float64 golden at small sizes;
  - about 3–5 s beyond process start.
- racecheck, memcheck and initcheck: run in parallel, about 8–15 s wall. They can be skipped when a value test
  already failed, because the revision goes back to the agent anyway.
- GPU-FPX: about 7 s, only on failing inputs or when the diff touches math.
- VOLTA: 1–15 s.
- **Total: about 20–35 s per revision**, against about 7 s today.
  - Over 10 iterations that adds a few minutes per optimization run.
  - The main saving available is a long-lived worker instead of one process per run, because process start
    dominates.

## 6. An evaluation protocol that avoids overfitting to the benchmark
1. **Freeze before measuring.**
   - Tag the commit of the design and the implementation.
   - Write down the families, ladders, budgets and checker settings.
   - After the tag, no rule, value or threshold changes before the held-out numbers are reported.
2. **A held-out set that nobody inspects.** Candidates:
   - new agent revisions from the Task 4 loop, logged automatically and never read before the run;
   - teammates' kernels;
   - at least one operator we did not design for, e.g. RMSNorm, or a GEMM with an epilogue. "Universal" is the
     claim, so the held-out set must include an operator the families were not tuned on.

   The current HELDOUT list is now partly a tuning set: regime thresholds and `spike` both came from it.
3. **Report tuning and held-out separately**, always. Include false alarms on correct kernels and on no-op edits,
   for every component.
4. **Count post-hoc rules.**
   - Keep a log of every rule, value or family added after looking at misses. It holds 2 so far: regimes, spike.
   - Report the count next to the numbers.
   - A rule added after the freeze resets the held-out set: new kernels must be collected.
5. **Ablations on the same mutants and the same held-out set:**
   - no parent oracle;
   - no search (coverage pass only);
   - no position classes (random positions);
   - no checkers;
   - no VOLTA;
   - no change routing: the same total budget spread uniformly, i.e. "generic tests always".

   The last one is the key comparison. It shows whether change-awareness helps beyond simply running more generic
   tests.
6. **Equal-budget baselines:**
   - fixed tests;
   - random inputs with the same number of launches;
   - KernelBench-Verified-style fixed distributions (x1, x3, x0.01, x−1).
7. **Decide equivalent mutants before running the tools.** Use VOLTA EQUIV or a written reason, so that a mutant
   no tool catches cannot be relabeled as harmless afterwards.

## 7. Open questions for the user and team
1. **Same-value writes.** In the 6 `if (lane == 0)` negations, VOLTA says RACE, racecheck says nothing and outputs
   never change. Do we count them as bugs? This decides 6 of the 14 "VOLTA-only" catches.
2. **Sanitizers in the pipeline.** They had no false alarms on this small set and cost about 8 s.
   - Should they become a validator now?
   - In which configuration? If B or C gets them, the A/B/C story changes. Possibly a new configuration "fixed +
     sanitizers" against "fixed + VOLTA".
3. **GPU-FPX: keep or drop?** It needs a third-party patch set ported to a newer NVBit than the one it targets.
   On this set it detected nothing that the value check missed. It added source lines for the two injected
   numerical bugs, but none for the GLM kernels. Is that worth the dependency?
4. **Held-out kernels.**
   - Can teammates provide kernels that nobody on the design side looks at?
   - Can the Task 4 agent runs be logged and frozen before anyone inspects them?
   - Which new operator do we use for the "universal" claim?
5. **Parent oracle and intended changes.** When the agent *means* to change numerics, e.g. a lower-precision
   accumulator, outputs that differ from the parent are expected. Should the agent declare its intent with each
   revision, so that we know when to expect differences?
6. **Budget.** What wall time per iteration is acceptable? The estimate is 20–35 s per revision, against about
   7 s now.
7. **Admissible inputs.** Is "PyTorch float32 passes on it" the right definition of a fair input? It needs the
   condition-aware tolerance for ops like LayerNorm with an offset.
8. **The uncommitted `spike` change.** Keep it in the floor until F2/F3 exist? Its v3 numbers (112/136, 56/67)
   should then be reported as "with a post-hoc rule".
9. **compute-sanitizer on tirx.** Should we try it? This needs one short GPU job.
