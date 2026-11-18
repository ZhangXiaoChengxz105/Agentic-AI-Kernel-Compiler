"""The agent feedback and repair loop (Task 4).

    for iteration 1..max_iters (default 10, the proposal's cap):
        messages = context.build_messages(...)          task + previous draft + its observation (+ best so far)
        reply    = llm.chat(messages)                   GLM-5.3 via Lumen (budget-guarded)
        cand     = Candidate(code, parent = last ACCEPTED candidate; the first is compared with the reference)
        result   = validation.run_pipeline(cand, config)          A0 | A | B | C: the same checks every iteration
        perf     = perf.measure(cand) if result.passed            speed, identical in every configuration
        observation = feedback.render_observation(result.reports, perf, best perf)
        log the iteration (trajectory.py)
        stop when the candidate passes (proposal: "until the candidate passes the applicable checks or reaches an
        iteration limit"); with optimize=True keep going for speed until the cap instead
    return the best correct candidate (fastest measured one; without a GPU, the first that passed)

A seed kernel (e.g. an injected bug from workloads/*/tirx/bugs/) can be validated as iteration 0 without a model
call; the loop then starts from its observation. This is how repair of a known-wrong kernel is measured.

Other stop reasons: the coin cap (max_coins, worst case checked before each call), a Lumen error, or an exception
inside the validators (a harness problem, not the agent's).

run_agent(workload, config="C", llm=client, ...) -> AgentResult
CLI: akc agent --workload softmax [--config C] [--max-iters 10] [--seed FILE] [--optimize] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..feedback.render import render_observation
from ..perf.report import PerfReport
from ..validation.interface import Candidate, ValidationContext
from ..validation.pipeline import CONFIGS, run_pipeline
from .context import build_messages
from .prompts import extract_code, system_prompt, task_prompt
from .trajectory import Step, Trajectory

MAX_ITERS = 10


@dataclass
class AgentResult:
    passed: bool                     # some candidate passed the configuration's checks
    best: Step | None                # best correct candidate (code, perf, label)
    steps: list[Step]
    stop_reason: str                 # passed | max_iters | budget | llm_error | validation_error | dry_run
    coins: float
    seconds: float
    out_dir: Path

    def summary(self) -> dict:
        llm_steps = [s for s in self.steps if s.llm is not None]
        first_pass = next((s.iteration for s in self.steps if s.passed), None)
        return {"passed": self.passed, "stop_reason": self.stop_reason,
                "best": self.best.label if self.best else None,
                "best_time_ms": (self.best.perf or {}).get("time_ms") if self.best else None,
                "best_speedup_vs_torch": (self.best.perf or {}).get("speedup_vs_torch") if self.best else None,
                "first_pass_iteration": first_pass, "iterations": len(llm_steps), "steps": len(self.steps),
                "coins": round(self.coins, 5), "seconds": round(self.seconds, 1),
                "validation_seconds": round(sum(s.seconds.get("validation", 0) for s in self.steps), 1),
                "llm_seconds": round(sum(s.seconds.get("llm", 0) for s in self.steps), 1)}


def _validate(step: Step, cand: Candidate, config: str, ctx: ValidationContext, best_perf: PerfReport | None,
              measure_perf: bool) -> dict:
    """Run the configuration (and speed if it passed); fill the step. Returns the pipeline result as a dict."""
    t0 = time.perf_counter()
    res = run_pipeline(cand, config, ctx)
    step.seconds["validation"] = round(time.perf_counter() - t0, 2)
    step.passed = res.passed
    step.per_validator, step.stopped_after = res.per_validator, res.stopped_after
    step.failed_validators = [r.validator for r in res.reports if not r.passed]
    step.error_kinds = sorted({f.kind for f in res.findings if f.severity == "error"})
    perf = None
    if res.passed and measure_perf:
        from ..perf import measure
        t1 = time.perf_counter()
        perf = measure(cand, device=ctx.device)
        step.seconds["perf"] = round(time.perf_counter() - t1, 2)
        step.perf = perf.to_dict()
    step.observation = render_observation(res.reports, perf, best_perf)
    return res.to_dict()


def run_agent(workload: str, config: str = "C", *, llm, max_iters: int = MAX_ITERS, optimize: bool = False,
              seed: str | Path | None = None, device: str = "auto", measure_perf: bool = True,
              max_coins: float = 2.0, max_tokens: int = 100000, reasoning_effort: str | None = "high",
              dry_run: bool = False, out_dir: Path | None = None, ctx_extra: dict | None = None) -> AgentResult:
    if config not in CONFIGS:
        raise ValueError(f"unknown configuration {config!r}; known: {', '.join(CONFIGS)}")
    t_start = time.perf_counter()
    system, task = system_prompt(), task_prompt(workload)
    ctx = ValidationContext(config=config, device=device, extra=dict(ctx_extra or {}))
    traj = Trajectory(workload, config, out_dir, settings={
        "model": getattr(llm, "model", None), "max_iters": max_iters, "optimize": optimize,
        "seed": str(seed) if seed else None, "device": device, "measure_perf": measure_perf,
        "max_coins": max_coins, "max_tokens": max_tokens, "reasoning_effort": reasoning_effort, "dry_run": dry_run})
    accepted: Candidate | None = None                # parent policy: the last candidate that passed
    best: Step | None = None
    best_perf: PerfReport | None = None
    last: Step | None = None
    spent, stop = 0.0, "max_iters"

    def finish_step(step: Step, cand: Candidate | None, validation: dict | None) -> bool:
        """Bookkeeping after validation; True when the loop should stop because the candidate passed."""
        nonlocal accepted, best, best_perf
        if step.passed and cand is not None:
            step.accepted, accepted = True, cand
            perf = PerfReport.from_dict(step.perf) if step.perf else None
            if best is None or (perf is not None and perf.faster_than(best_perf)):
                step.best, best, best_perf = True, step, perf
        traj.record(step, validation)
        return step.passed and not optimize

    if seed is not None:                             # iteration 0: a given kernel, no model call
        code = Path(seed).read_text()
        step = Step(0, "seed", "reference", code=code)
        cand = Candidate(workload, "tirx", code, label="seed")
        try:
            validation = _validate(step, cand, config, ctx, best_perf, measure_perf)
        except Exception as e:  # noqa: BLE001  (a validator crashed: harness problem)
            step.observation = f"internal validation error: {type(e).__name__}: {e}"
            traj.record(step)
            return _done(traj, best, "validation_error", spent, t_start)
        step.seconds["total"] = step.seconds.get("validation", 0) + step.seconds.get("perf", 0)
        if finish_step(step, cand, validation):
            return _done(traj, best, "passed", spent, t_start)
        last = step

    for it in range(1, max_iters + 1):
        t_it = time.perf_counter()
        msgs = build_messages(system, task, traj.steps, last, best, it, max_iters, optimize)
        traj.prompt(it, msgs)
        if hasattr(llm, "price"):                    # whole-run coin cap (the client also guards each call)
            p_in, p_out = llm.price()
            worst = llm.estimate_tokens(msgs) / 1e6 * p_in + max_tokens / 1e6 * p_out
            if spent + worst > max_coins:
                stop = "budget"
                break
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        try:
            res = llm.chat(msgs, max_tokens=max_tokens, dry_run=dry_run, tag=f"agent-{workload}-{config}-it{it}",
                           **extra)
        except Exception as e:  # noqa: BLE001  (LumenError, BudgetError, network)
            traj.record(Step(it, f"it{it}", accepted.label if accepted else "reference",
                             observation=f"model call failed: {type(e).__name__}: {e}"))
            stop = "llm_error"
            break
        spent += res.cost
        traj.reply(it, res)
        llm_info = {"model": res.model, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "cost": round(res.cost, 6), "latency_s": round(res.latency_s, 1), "finish": res.finish_reason}
        if res.dry_run:
            traj.record(Step(it, f"it{it}", "reference", observation="dry run: request not sent", llm=llm_info))
            stop = "dry_run"
            break
        label = f"it{it}"
        step = Step(it, label, accepted.label if accepted else "reference", code=extract_code(res.content),
                    llm=llm_info, seconds={"llm": round(res.latency_s, 1)})
        validation, cand = None, None
        if step.code is None:
            step.observation = ("your answer hit the output length limit before any code was written."
                                if res.finish_reason == "length" else "your answer had no fenced code block.")
        else:
            cand = Candidate(workload, "tirx", step.code, parent=accepted, label=label)
            try:
                validation = _validate(step, cand, config, ctx, best_perf, measure_perf)
            except Exception as e:  # noqa: BLE001
                step.observation = f"internal validation error: {type(e).__name__}: {e}"
                traj.record(step)
                stop = "validation_error"
                break
        step.seconds["total"] = round(time.perf_counter() - t_it, 2)
        if finish_step(step, cand, validation):
            stop = "passed"
            break
        last = step
    return _done(traj, best, stop, spent, t_start)


def _done(traj: Trajectory, best: Step | None, stop: str, spent: float, t_start: float) -> AgentResult:
    out = AgentResult(best is not None, best, traj.steps, stop, spent, time.perf_counter() - t_start, traj.dir)
    traj.finish(out.summary(), best)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, help="softmax | layernorm | gemm")
    ap.add_argument("--config", default="C", choices=list(CONFIGS))
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS, help="model calls per run (proposal: 10)")
    ap.add_argument("--optimize", action="store_true",
                    help="keep improving speed after a pass until --max-iters (default: stop at the first pass)")
    ap.add_argument("--seed", help="kernel module validated as iteration 0 (e.g. an injected bug to repair)")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--no-perf", action="store_true", help="do not measure speed of passing candidates")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--max-coins", type=float, default=2.0, help="hard cap for the whole run")
    ap.add_argument("--per-call-limit", type=float, default=1.0, help="Lumen client guard per call (coins)")
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "none"])
    ap.add_argument("--http-timeout", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true", help="build and log the first prompt; send nothing")
    ap.add_argument("--out", help="run directory (default results/agent/<timestamp>_<workload>_<config>)")
    a = ap.parse_args(argv)
    from ..llm.lumen import LumenClient
    client = LumenClient(model=a.model, per_call_limit=a.per_call_limit, timeout_s=a.http_timeout)
    r = run_agent(a.workload, a.config, llm=client, max_iters=a.max_iters, optimize=a.optimize, seed=a.seed,
                  device=a.device, measure_perf=not a.no_perf, max_coins=a.max_coins, max_tokens=a.max_tokens,
                  reasoning_effort=None if a.reasoning_effort == "none" else a.reasoning_effort,
                  dry_run=a.dry_run, out_dir=Path(a.out) if a.out else None)
    for s in r.steps:
        verdict = "PASS" if s.passed else ("fail" if s.code is not None else "-")
        print(f"{s.label:6s} {verdict:4s} {', '.join(s.failed_validators or s.error_kinds):30s} "
              f"{s.seconds}")
    print(f"stop: {r.stop_reason}; best: {r.best.label if r.best else '-'}; coins {r.coins:.4f}; {r.out_dir}")
    return 0 if r.passed or r.stop_reason == "dry_run" else 1


if __name__ == "__main__":
    sys.exit(main())
