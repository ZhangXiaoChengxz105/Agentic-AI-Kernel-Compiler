# Project proposal (source of truth)

Pasted by the team on 2026-11-17. PDF-extraction spacing was fixed; the wording is unchanged.

**The only allowed difference:** CAKE has no public code, so tirx-lite (MLC TIRx-harness) plays the role of CAKE IR plus its compiler (see README §2 "CAKE substitute"). Everything else should follow this text.

Consequences:
- the agent writes tirx-lite only;
- change analysis works on IR operations;
- "CAKE's existing compiler checks" run first;
- VOLTA checks the compiled kernels.

Hand-written CUDA kernels in `workloads/*/cuda` are legacy/extras and are not part of configurations A/B/C.

---

## Design an Agentic AI Kernel Compiler
Hongyi Chen (hongyic5@illinois.edu), Minghao Fang (mf46@illinois.edu), Xiaocheng Zhang (xz105@illinois.edu)

### 1 Problem Description
AI agents can generate and optimize GPU kernels for machine learning workloads. However, ensuring that optimized
GPU kernels remain semantically equivalent to their reference implementations and free of data races and deadlocks
is a key challenge. Passing a fixed test set does not guarantee that a generated kernel correctly implements its
intended computation, especially when optimizations introduce errors such as incorrect boundary handling after
tiling changes or synchronization errors after instruction reordering.

CAKE uses compiler–agent co-design: agents express kernel implementations in CAKE IR, while the compiler checks the
IR, generates lower-level code, and provides diagnostics to support refinement. Building on this workflow, we will
extend the validation and feedback mechanisms with VOLTA verification and change-aware testing. VOLTA checks
semantic equivalence within its supported scope and real-number model, and change-aware testing can target
potential errors associated with the agent's modifications. The agent will use verification diagnostics and test
failures to revise the kernel implementation.

We will study a small set of kernels supported by VOLTA and compare three configurations: a fixed set of test
cases, the same tests with VOLTA verification, and the tests augmented with both VOLTA and dynamically generated
change-aware tests. We will evaluate verification coverage, validation overhead, and final kernel performance. If
erroneous candidates are available, we will also assess error detection and repair success.

### 2 Proposed Implementation
We will extend the CAKE kernel-generation loop with two complementary validation mechanisms: change-aware testing
and VOLTA verification.

First, we will implement a change-aware test generator. After the agent revises a Cake IR kernel, we will inspect
the modified IR operations and classify the change into a small set of categories such as tiling, memory access,
synchronization, or numerical computation. Based on the change type, the system will generate targeted test inputs
likely to expose errors introduced by that modification. For example, changes to tiling or indexing will trigger
boundary and non-divisible shapes, while synchronization changes will trigger tests stressing different pipeline
stages and execution sizes.

Second, we will integrate VOLTA as an additional semantic verification step. Candidates that pass CAKE's existing
compiler checks will be checked by VOLTA when the kernel is within VOLTA's supported scope. Verification failures
and counterexamples, together with failures from the generated tests, will be converted into textual diagnostics
and returned to the optimization agent. The agent will then revise the Cake IR kernel and repeat the validation
loop.

### 3 Tasks and Estimated Time
We plan to complete the project in approximately 5–5.5 weeks. The implementation will be divided into five tasks,
with some tasks carried out in parallel.

**Task 1: Environment Setup and Baseline (Week 1).** Set up the CAKE compilation and optimization pipeline. Select
2–3 representative GPU kernels with reference implementations and establish baseline correctness tests and
performance measurements. Confirm that the selected kernels can be analyzed by VOLTA.

**Task 2: Change-Aware Test Generation (Weeks 2–3).** Implement an IR change analysis module that identifies
modifications to tiling, memory access, synchronization, and numerical operations. Develop a test generator that
produces targeted inputs based on the detected changes, including boundary cases, non-divisible dimensions, and
numerical edge cases.

**Task 3: VOLTA Verification Integration (Weeks 2–3).** Integrate VOLTA into the kernel validation pipeline.
Implement the interface for invoking verification on supported compiled kernels and collecting verification
results, counterexamples, and diagnostic information. Convert these results into structured feedback for the
optimization agent.

**Task 4: Agent Feedback and Repair Loop (Week 4).** Connect change-aware testing and VOLTA verification to the
agent-driven kernel optimization loop. Implement an iterative workflow in which the agent receives validation
feedback, revises the kernel, and repeats validation until the candidate passes the applicable checks or reaches
an iteration limit.

**Task 5: Evaluation and Analysis (Weeks 5–5.5).** Evaluate three configurations: fixed tests only, fixed tests
with VOLTA, and fixed tests with VOLTA and change-aware testing. Compare error detection, verification coverage,
validation overhead, and final kernel performance. Where erroneous candidates are available, measure the agent's
ability to detect and repair them.
