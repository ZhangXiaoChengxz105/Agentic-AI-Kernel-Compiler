# Related tools for change-aware testing of GPU kernels (survey, 2026-11-17)

**Where the facts come from.** Only abstracts, project pages, READMEs and search-result summaries. We did not read
the full papers. Each item says what is "confirmed" (stated in an abstract, README or project page) and what comes
only from a search summary. Two exceptions, checked by us on Delta: compute-sanitizer's version and path, and
whether GPU-FPX builds. See `notes/gpu_results.md`, section "Operator-agnostic checkers".

Each item gives:
- what it does;
- how it relates to us;
- whether code is available.

## (a) Change-based test generation

- **Just-in-Time Catching Test Generation at Meta** (Becker, Harman et al., arXiv 2601.22832, Jan 2026).
  https://arxiv.org/abs/2601.22832
  - What it does: generates "catching" tests for a code change (diff) before it lands.
    - A catching test passes on the parent revision and fails on the diff.
    - Confirmed from the abstract: change-aware methods give about 4x more candidate catches than "hardening"
      tests and 20x more than coincidental failures; 22,126 generated tests; LLM and rule-based assessors cut human
      review by 70%; 41 catches reported, 8 true positives.
    - Search summary of the full text only: an LLM infers the risks of the diff, and the risks become mutants of
      the parent.
  - Relation to us: closest in spirit. Like us, it asks what this revision could break and tests that, rather than
    re-running a fixed suite. It targets server-side code, not GPU kernels, and judges tests by parent-vs-diff
    behaviour.
  - Code: internal Meta system; no public code found.
- **Harden and Catch for Just-in-Time Assured LLM-Based Software Testing** (Harman, O'Hearn, Sengupta;
  FSE 2025 keynote paper; arXiv 2504.16472). https://arxiv.org/abs/2504.16472
  - What it does: a position paper. It defines hardening tests (protect future changes) and catching tests (find
    bugs in this change), and poses the "JiTTest" challenge. The Meta paper above builds on it.
  - Relation to us: it names our idea. Our change-aware tests are catching tests; the fixed suite is a hardening
    suite.
  - Code: none (position paper).
- **KATCH** (Marinescu, Cadar; ESEC/FSE 2013). https://srg.doc.ic.ac.uk/projects/katch/,
  DOI 10.1145/2491411.2491438
  - What it does: targeted symbolic execution on KLEE. It steers the search toward the lines a patch changed, to
    reach and test them. Programs are C, run on the CPU.
  - Relation to us: the same "go where the patch is" idea, by symbolic search instead of our rules.
  - Code: source tarball and VM on the project page (FSE 2013 artifact award). No license stated.
- **Shadow symbolic execution** (Kuchta, Palikareva, Cadar; ACM TOSEM 27(3), 2018, DOI 10.1145/3208952; ICSE 2016
  version: "Shadow of a Doubt: Testing for Divergences Between Software Versions").
  https://srg.doc.ic.ac.uk/projects/shadow/
  - What it does: runs the old and new versions together in one symbolic execution and generates inputs on which
    they diverge.
  - Relation to us: this is differential testing against the parent plus a search for diverging inputs. Our
    "compare the revision with its parent and the reference" step is the testing half of it, without the symbolic
    search.
  - Code: VM with binaries only ("contact the authors"); no public source found.
- **Shared idea of (a):**
  - differential testing against the parent revision;
  - a search for inputs on which the two diverge;
  - the change decides where to search.

  None of them targets GPU kernels or floating-point behaviour.

## (b) Operator-agnostic GPU checkers

- **NVIDIA compute-sanitizer.** https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
  - What it does: runtime checking of unmodified CUDA programs. It has four tools:
    - memcheck: out-of-bounds and misaligned global/shared/local accesses, plus API errors;
    - racecheck: shared-memory hazards **only**. The docs say it "only supports detecting accesses to on-chip
      shared memory"; release notes up to 2026 add only shared-side features such as cluster shared memory,
      `cp.async.bulk` and deadlock detection;
    - initcheck: reads of uninitialized global memory, or shared memory with `--initcheck-address-space`;
    - synccheck: invalid use of `__syncthreads` / `__syncwarp`, such as divergent barriers.
  - On Delta (verified by us): version 2026.1.1 at
    `/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/compute-sanitizer/compute-sanitizer` after
    `module load cudatoolkit/26.5_13.2`.
  - Relation to us: it is a candidate replacement for part of what only VOLTA catches today (shared-memory races
    and out-of-bounds accesses), and it works on any CUDA kernel, not just VOLTA's scope. It is dynamic: it only
    sees what the given input executes. See our trial in `notes/gpu_results.md`.
  - Code: closed source, free, ships with the CUDA toolkit.
- **GPU-FPX** (Li, Laguna, Fang, Swirydowicz, Li, Gopalakrishnan; HPDC 2023, DOI 10.1145/3588195.3592991).
  https://github.com/LLNL/GPU-FPX
  - What it does: NVBit binary instrumentation, loaded with `LD_PRELOAD` into unmodified programs.
    - The detector reports, per kernel and source line, where NaN, INF, subnormal or division-by-zero results
      occur.
    - The analyzer shows whether an exceptional value appears at an instruction or only flows through it.
  - Relation to us: an operator-agnostic numerical checker. It reports an INF in an intermediate value even when
    the final output hides it. Like every dynamic tool, it only fires if the input triggers the exception.
  - Code: MIT. It is shipped as patches on NVBit 1.5.5's `record_reg_vals` sample, and the README does not state
    which CUDA versions it supports.
  - Verified by us:
    - NVBit 1.5.5 requires a driver ≤ 510, so we built the patched sources against NVBit 1.8 with three small API
      changes;
    - it runs on Delta (driver 595, A100);
    - in our trial it reported INF/NaN for the bugs whose inputs trigger them, but also INF in correct
      online-softmax kernels.
- **Xscope** (Laguna, Gopalakrishnan; SC22, "Finding Inputs that Trigger Floating-Point Exceptions in GPUs via
  Bayesian Optimization"). https://github.com/LLNL/Xscope
  - What it does: Bayesian optimization over input values to find inputs that make a GPU function raise
    floating-point exceptions.
    - Confirmed: about 72% of 58 CUDA math functions. The journal version (Parallel Computing 2023) reports about
      73%.
    - The claim that the original handles at most 3 inputs was **not confirmed** in any abstract or README we
      found.
  - Extension: **FPBoxer** (Tran, Laguna, Gopalakrishnan; HPDC 2024, DOI 10.1145/3625549.3658660). A search
    summary says it handles more than 20 inputs with about a 90x speedup. No code found.
  - Relation to us: input search for exceptional values is the generic version of our hand-made `spike` rows. It
    searches for values instead of hard-coding them.
  - Code: Xscope is public. The README says "MPI license", probably a typo; the GitHub sidebar says MIT.
- **Faial** (Cogumbreiro, Lange, Liew, Zicarelli; CAV 2021, "Checking Data-Race Freedom of GPU Kernels,
  Compositionally", DOI 10.1007/978-3-030-81685-8_19; extended in FMSD 2023 as "Memory access protocols:
  certified data-race freedom for GPU kernels"). https://gitlab.com/umb-svl/faial
  - What it does: static data-race-freedom checking of CUDA kernels.
    - Confirmed: 218 of 227 kernels verified.
    - "No annotations needed" comes from a search summary. A later study found that about 27% of race-free
      programs need user assertions.
  - Relation to us: a static, input-independent race checker, like VOLTA's race check, that would also cover
    kernels outside VOLTA's equivalence scope.
  - Code: MIT, OCaml, with Docker image and binaries.
- **GPUVerify** (Betts, Chong, Donaldson, Qadeer, Thomson; OOPSLA 2012, DOI 10.1145/2384616.2384625).
  https://multicore.doc.ic.ac.uk/tools/GPUVerify/
  - What it does: static verification of race- and barrier-divergence-freedom for OpenCL and CUDA kernels.
  - Relation to us: the same role as Faial; older and well known.
  - Code: https://github.com/mc-imperial/gpuverify (Ms-PL).

## (c) Correctness of LLM-written kernels

- **KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?** (arXiv 2607.16241, Jun 2026).
  https://arxiv.org/abs/2607.16241
  - What it does: re-checks KernelBench results with a hidden test suite over 4 input distributions.
    - Confirmed from the abstract: hard-coded outputs and an unfair TF32 baseline; the best model gets 0.88x
      instead of the claimed 1.43x speedup.
    - From a full-text summary only: "skipped computation"; the distributions are input scale x1, x3, x0.01 and
      x-1.
  - Relation to us: the same lesson as our fixed suite. Typical inputs let wrong kernels pass, and more varied
    inputs catch them. Their distributions are fixed, not chosen per change.
  - Code: https://github.com/facebookresearch/kernel_bench_verified (MIT).
- **A Contract-Grade Verifier for LLM-Generated GPU Kernels, and a Native Blackwell Backward for the
  Gated-Linear-Recurrence Family** (Shah, Shrestha; arXiv 2608.12700). https://arxiv.org/abs/2608.12700
  - What it does: 12 adversarial "gates", several tolerance-free.
    - The gates check NaN/INF masking, run-to-run nondeterminism, shape changes, and fp16 accumulation against an
      fp32 reference.
    - Confirmed from the abstract: on 2,638 kernels, 39.5% broken and 62.1% with at least one violation.
  - Relation to us: the closest list of generic checks.
    - Our 2 sub-tolerance tirx misses include a float16 sum accumulator, exactly their fp16-accumulation gate.
    - Their nondeterminism gate matches our repeat-and-compare check.
  - Code: https://github.com/RishiShah99/lethe (MIT), linked from Zenodo 10.5281/zenodo.21563213 rather than from
    arXiv. We have not checked that it runs.
- **Kuiper** (PLDI 2026, PACMPL 10, DOI 10.1145/3808280, "Correct and Efficient GPU Programming with Dependent
  Types and Separation Logic"), and the workshop paper **"The Next Frontier for AI-Generated Kernels:
  Correctness"** (Martínez, Sorensen; PLDI 2026 PAgE).
  - What they do: kernels written in F*/Pulse with proofs of memory safety, race freedom and functional
    correctness. The workshop paper reports verified implementations of all 100 KernelBench Level 1 tasks, mostly
    written by LLM agents.
  - Relation to us: full proofs instead of testing. The kernel must be written in their language, so it does not
    check existing CUDA.
  - Code: https://github.com/FStarLang/kuiper (license not shown).
- **ProofWright** (Chatterjee et al., NVIDIA / Stanford; arXiv 2511.12294). https://arxiv.org/abs/2511.12294
  - What it does: VerCors for memory and thread safety, Rocq for equivalence.
    - Confirmed: safety verified for 74% of KernelBench L1 kernels.
    - The "14% semantic equivalence" figure appears only in a search summary. The abstract says equivalence is
      shown for "a class of element-wise kernels".
  - Relation to us: VOLTA's role (equivalence plus races) on a different toolchain; the same scope limits.
  - Code: no link found.
- **robust-kbench** (Sakana AI, "Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and
  Optimization", arXiv 2509.14279, Sep 2025). https://arxiv.org/abs/2509.14279
  - What it does: a more robust KernelBench variant. It uses LLM-based verifiers to judge and filter candidate
    kernels.
  - Relation to us: an LLM as a correctness judge. We use rules plus tests and VOLTA, and an LLM only as an
    optional classifier.
  - Code: https://github.com/SakanaAI/robust-kbench (Apache-2.0). The README does not describe the verifier.

## (d) ML compiler testing

- **NNSmith** (Liu et al., ASPLOS 2023, arXiv 2207.13066). https://github.com/ise-uiuc/nnsmith
  - What it does: generates random valid DNN models and compares results across compilers (differential
    testing).
    - It uses gradient-based input search to **avoid** NaN/INF, so that differences are real bugs rather than
      float noise.
    - The abstract says 72 new bugs (58 confirmed, 51 fixed). The number 65 in our task brief does not match the
      abstract.
  - Relation to us: differential testing plus input search to steer away from false alarms. Our tolerance and
  float64-golden design has the same aim.
  - Code: Apache-2.0.
- **Tzer** (Liu et al., OOPSLA 2022, arXiv 2202.09947). https://github.com/ise-uiuc/tzer
  - What it does: coverage-guided fuzzing of TVM by jointly mutating the low-level IR and the pass sequence. The
    abstract reports 49 bugs (37 confirmed, 25 fixed).
  - Relation to us: mutation of IR. Our mutation benchmark (`akc mutants`) mutates kernels to measure detection;
    Tzer mutates IR to find compiler bugs.
  - Code: Apache-2.0.

## (e) Python property-based testing

- **Hypothesis**, including `hypothesis.extra.numpy`. https://github.com/HypothesisWorks/hypothesis (MPL-2.0).
  Docs: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#hypothesis-numpy
  - What it does:
    - It generates inputs from strategies: `arrays`, `array_shapes`, `from_dtype`. Its `floats()` produces NaN,
      inf and subnormals unless bounds exclude them.
    - It shrinks a failing case to a minimal one.
    - It stores failing examples and replays them.
  - Relation to us: a ready-made engine for "input families plus search plus shrinking".
    - Generic families such as sparse extremes, boundary sizes or row-scale alternation could be strategies.
    - A shrunk counterexample, e.g. the smallest row and the single position of the outlier, makes much better
      agent feedback than a 64x1000 tensor.
  - Code: open source, pip-installable.

## What this means for us

- We found no existing tool that generates tests per revision, from the change, for GPU kernels. Meta's JiT
  catching tests come closest, but they are internal and target ordinary code.
- Every part of our design has a general tool:
  - differential testing against the parent and the reference (Shadow, NNSmith);
  - targeting the changed code (KATCH, Meta JiT);
  - searching inputs for floating-point exceptions (Xscope / FPBoxer, Hypothesis);
  - operator-agnostic runtime checkers for memory, races and float exceptions (compute-sanitizer, GPU-FPX);
  - static race and equivalence proofs (VOLTA, Faial, GPUVerify, ProofWright, Kuiper);
  - generic adversarial checks for LLM kernels (the contract-grade verifier, KernelBench-Verified).
- So the change-aware part should not invent value distributions per operator. It should:
  - combine these general mechanisms;
  - use the diff to decide where to look (which shapes, which code paths, how much search budget).

  `notes/design_universal_tests.md` works this out, using the trial in `notes/gpu_results.md`.
