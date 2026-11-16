"""The agent: model + context + loop + trajectory (Task 4 builds the full loop here).

    pilot.py       working prototype: prompt -> GLM -> code -> validator -> rendered feedback, <= N rounds,
                   correctness only (results in notes/glm_pilot.md)
Planned (see docs/architecture.md):
    prompts/       system prompts, task templates per language, constraints and known pitfalls
    context.py     what the model sees each turn (task + best code so far + compact feedback + perf numbers)
    loop.py        controller: generate -> validation.run_pipeline(config) -> feedback.render -> revise;
                   max 10 iterations; keep the fastest candidate that passes; parent = last accepted candidate
    trajectory.py  per-iteration record (prompt, reply, code, reports, timings, coins) for evaluation
"""
