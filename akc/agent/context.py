"""What the model sees each iteration (the agent's context).

The context is rebuilt every turn instead of growing a chat history: GLM has no prompt-cache discount, so
every resent token costs coins, and old failing drafts only distract. Each turn sends
    system   the system prompt
    user     the task prompt (prompts.task_prompt)
    then, from the second iteration on:
    user     a progress table (one line per earlier iteration: verdict, failing validators, time),
             the previous draft and its observation (feedback.render_observation: validation, then speed),
             and, when it differs from the previous draft, the best correct version so far with its speed,
             followed by the instruction for this turn (repair, or make it faster while staying correct).

build_messages(system, task, history, last, best, iteration, max_iters, optimize) -> list[dict]
`history` / `last` / `best` are Step objects from loop.py (only the fields used below are read).
"""
from __future__ import annotations

REPAIR = ("Fix every problem listed above and return the complete module. Keep what already works; change only "
          "what the findings point to unless a redesign is needed.")
FASTER = ("This version passed every check. Make it faster at the benchmark shape while staying correct on the "
          "whole shape range, and return the complete module.")
NO_CODE = "Return the complete module in one ```python code block."


def _status(step) -> str:
    if step.code is None:
        return "no code in the reply"
    if step.passed:
        t = step.perf.get("time_ms") if step.perf else None
        return "passed" + (f", {t:.4f} ms" if t else "")
    return "FAILED (" + ", ".join(step.failed_validators or ["?"]) + ")"


def progress(history) -> str:
    return "\n".join(f"- {s.label}: {_status(s)}" for s in history)


def build_messages(system: str, task: str, history: list, last, best, iteration: int, max_iters: int,
                   optimize: bool) -> list[dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    if last is None:
        return msgs
    parts = [f"Iteration {iteration} of {max_iters}. Earlier versions:\n{progress(history)}"]
    if last.code is None:
        parts.append(f"Your previous answer ({last.label}): {last.observation}")
    else:
        parts.append(f"=== Previous version ({last.label}) ===\n```python\n{last.code.rstrip()}\n```\n\n"
                     f"=== What our checks found for {last.label} ===\n{last.observation}")
    if best is not None and best is not last:
        speed = ""
        if best.perf and best.perf.get("time_ms"):
            x = best.perf.get("speedup_vs_torch")
            speed = f", {best.perf['time_ms']:.4f} ms" + (f" = {x:.2f}x PyTorch" if x else "")
        parts.append(f"=== Best correct version so far ({best.label}{speed}) ===\n"
                     f"```python\n{best.code.rstrip()}\n```")
    if last.code is None:
        parts.append(NO_CODE)
    elif last.passed and optimize:
        parts.append(FASTER)
    else:
        parts.append(REPAIR)
    msgs.append({"role": "user", "content": "\n\n".join(parts)})
    return msgs
