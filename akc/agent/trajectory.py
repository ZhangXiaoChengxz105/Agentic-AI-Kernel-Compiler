"""Per-iteration record of an agent run (logged every iteration, read by the evaluation in Task 5).

Step        one iteration: the prompt, the reply, the code, the validation result, the speed, the observation,
            timings and coins. Iteration 0 is an optional seed kernel given by the user (no model call).
Trajectory  writes a run directory as it goes, so a crash or a budget stop keeps everything done so far:
    results/agent/<timestamp>_<workload>_<config>/
        run.json               settings of the run (workload, config, model, limits, seed)
        trajectory.jsonl       one line per iteration (Step.summary(): verdicts, per-validator seconds, coins, ...)
        it<k>_prompt.json      the exact messages sent to the model
        it<k>_reply.md         the reply (reasoning + content)
        it<k>.py               the extracted kernel module
        it<k>_validation.json  run_pipeline(...).to_dict() (every report with its findings and artifacts)
        it<k>_observation.txt  the feedback text the model got next
        best.py                the best correct kernel (when one passed)
        summary.json           outcome: passed, best iteration, stop reason, iterations, coins, seconds
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..paths import RESULTS


@dataclass
class Step:
    iteration: int
    label: str
    parent: str                              # label of the last accepted candidate, or "reference"
    code: str | None = None                  # None: the reply had no code block
    passed: bool = False                     # validation verdict of the configuration
    accepted: bool = False                   # became the parent of later candidates (passed)
    best: bool = False                       # best correct version at this point
    failed_validators: list[str] = field(default_factory=list)
    error_kinds: list[str] = field(default_factory=list)
    per_validator: dict = field(default_factory=dict)
    stopped_after: str | None = None
    perf: dict | None = None                 # PerfReport.to_dict() (only for candidates that passed)
    observation: str = ""
    llm: dict | None = None                  # model, tokens, cost, latency, finish reason (None for the seed)
    seconds: dict = field(default_factory=dict)   # llm / validation / perf / total

    def summary(self) -> dict:
        p = self.perf or {}
        return {"iteration": self.iteration, "label": self.label, "parent": self.parent,
                "has_code": self.code is not None, "passed": self.passed, "accepted": self.accepted,
                "best": self.best, "failed_validators": self.failed_validators, "error_kinds": self.error_kinds,
                "stopped_after": self.stopped_after, "per_validator": self.per_validator,
                "perf": {k: p.get(k) for k in ("measured", "reason", "time_ms", "speedup_vs_torch", "shape")}
                if self.perf else None,
                "llm": self.llm, "seconds": self.seconds}


class Trajectory:
    def __init__(self, workload: str, config: str, out_dir: Path | None = None, settings: dict | None = None):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(out_dir) if out_dir else RESULTS / "agent" / f"{stamp}_{workload}_{config}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[Step] = []
        (self.dir / "run.json").write_text(json.dumps({"workload": workload, "config": config, "started": stamp,
                                                       **(settings or {})}, indent=1, default=str))

    def _write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text)

    def prompt(self, it: int, messages: list[dict]) -> None:
        self._write(f"it{it}_prompt.json", json.dumps(messages, indent=1))

    def reply(self, it: int, res) -> None:
        self._write(f"it{it}_reply.md", f"# iteration {it}\n\nmodel={res.model} in={res.input_tokens} "
                    f"out={res.output_tokens} cost={res.cost:.5f} latency={res.latency_s:.1f}s "
                    f"finish={res.finish_reason}\n\n## reasoning\n{res.reasoning}\n\n## content\n{res.content}\n")

    def record(self, step: Step, validation: dict | None = None) -> None:
        it = step.iteration
        if step.code is not None:
            self._write(f"it{it}.py", step.code)
        if validation is not None:
            self._write(f"it{it}_validation.json", json.dumps(validation, indent=1, default=str))
        self._write(f"it{it}_observation.txt", step.observation + "\n")
        with open(self.dir / "trajectory.jsonl", "a") as f:
            f.write(json.dumps(step.summary(), default=str) + "\n")
        self.steps.append(step)

    def finish(self, summary: dict, best: Step | None) -> None:
        if best is not None:
            self._write("best.py", best.code)
        self._write("summary.json", json.dumps(summary, indent=1, default=str))
