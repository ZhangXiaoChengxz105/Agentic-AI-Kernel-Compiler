"""akc: Agentic Kernel Compiler (CS 521).

An LLM agent writes GPU kernels; validators check them; feedback drives revisions.

    task/        what to solve: workloads (manifests), PyTorch references, input distributions, TestCase
    llm/         model access (Lumen)
    agent/       the agent: prompts, (context, loop, trajectory: Task 4), pilot prototype
    backend/     compile (CUDA -> PTX, tirx-lite), launch on the GPU, time kernels
    validation/  validators with one interface (interface.py): fixed tests, VOLTA, change-aware tests;
                 pipeline.py composes them into configurations A / B / C
    feedback/    validation findings -> text the model can act on
    evaluation/  experiments over workloads and bug sets
See docs/architecture.md.
"""
