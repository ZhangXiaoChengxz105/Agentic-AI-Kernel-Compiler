"""Configurations A / B / C = which validators run, in which order (proposal section 1).

    A: fixed tests
    B: fixed tests + VOLTA (workload default shape)
    C: fixed tests + change-aware tests + VOLTA (default + change-aware shapes)

run_pipeline(cand, config, ctx) -> PipelineResult
  * a compile error stops the pipeline (later validators would only repeat it)
  * in C, the change-aware report's `suggested_volta_shapes` become VOLTA's shapes
PipelineResult.passed = no error finding in any report; .decided_by tells which validators reached a verdict.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .changeaware.validator import ChangeAwareValidator
from .fixed_tests.validator import FixedTestsValidator
from .interface import Candidate, Finding, ValidationContext, ValidationReport
from .volta.validator import VoltaValidator

CONFIGS = {"A": ("fixed_tests",), "B": ("fixed_tests", "volta"), "C": ("fixed_tests", "changeaware", "volta")}


@dataclass
class PipelineResult:
    config: str
    reports: list[ValidationReport] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.reports for f in r.findings]

    @property
    def decided_by(self) -> dict[str, bool]:
        return {r.validator: r.decided for r in self.reports}

    def to_dict(self) -> dict:
        return {"config": self.config, "passed": self.passed, "seconds": self.seconds,
                "reports": [r.to_dict() for r in self.reports]}


def run_pipeline(cand: Candidate, config: str = "C", ctx: ValidationContext | None = None,
                 stop_on_compile_error: bool = True) -> PipelineResult:
    ctx = (ctx or ValidationContext()).with_(config=config)
    res, t0, suggested = PipelineResult(config), time.perf_counter(), None
    for name in CONFIGS[config]:
        if name == "fixed_tests":
            rep = FixedTestsValidator().validate(cand, ctx.with_(test_cases=None))
        elif name == "changeaware":
            rep = ChangeAwareValidator().validate(cand, ctx)
            suggested = rep.artifacts.get("suggested_volta_shapes")
        else:
            rep = VoltaValidator().validate(cand, ctx.with_(volta_shapes=suggested if config == "C" else None))
        res.reports.append(rep)
        if stop_on_compile_error and any(f.kind == "compile_error" for f in rep.findings):
            break
    res.seconds = time.perf_counter() - t0
    return res
