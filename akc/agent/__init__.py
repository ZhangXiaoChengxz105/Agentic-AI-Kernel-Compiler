"""The agent: model + context + loop + trajectory (Task 4).

    prompts.py     system prompt and the tirx-lite task prompt for any workload (from its manifest), pitfalls
    context.py     what the model sees each turn (task + previous draft + its observation + best code so far)
    loop.py        controller: generate -> validation.run_pipeline(config) -> feedback.render_observation -> revise;
                   max 10 iterations; stop at the first pass (or keep the fastest with --optimize);
                   parent = last accepted candidate; `akc agent`
    trajectory.py  per-iteration record (prompt, reply, code, reports, perf, timings, coins) for evaluation
    pilot.py       the earlier prototype (softmax only, correctness only; results in notes/glm_pilot.md)
"""
