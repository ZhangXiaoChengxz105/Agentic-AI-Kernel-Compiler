"""Configurations = which validators run, in which order (proposal sections 1-2; the gate is D2).

    A0: fixed tests                                         ablation without the gate (= the old configuration A)
    A : compiler checks + fixed tests                       the CAKE baseline: its existing compiler checks + tests
    B : compiler checks + fixed tests + VOLTA               VOLTA at the workload default shape
    C : compiler checks + fixed tests + change-aware tests + VOLTA (at the default + change-aware shapes)

run_pipeline(cand, config="C", ctx=None, stop_on_compile_error=True, cache=None, skip=()) -> PipelineResult
Stop rules:
  * a compile_error finding (a static error of the gate, or a build failure seen by a later validator) always stops
    the pipeline: the later validators could only repeat it;
  * any other failing validator (e.g. a sanitizer race) also stops it by default, so the agent loop gets the first
    failure quickly; with ctx.extra["run_all"] = True the later validators still run (the evaluation sets it, so each
    validator's catch and seconds can be counted);
  * when the gate is part of the configuration, the NumSim test runner skips Racecheck (on a CPU-only node the gate
    already ran NumSim racecheck / synccheck): ctx.extra["numsim_racecheck"] = False for the test validators;
  * in C, the change-aware report's `suggested_volta_shapes` become VOLTA's shapes.
cache: a dict shared by the calls for ONE candidate and context (e.g. A0, A, B and C of one bug in the evaluation).
  A validator whose inputs are the same (name, candidate and parent source, device, VOLTA shapes, NumSim-racecheck
  switch, the relevant ctx.extra keys) is not run again: its report, with its own measured seconds, is reused.
skip: validator names left out on purpose (e.g. ("volta",) for a tests-only detection run); listed in .skipped.
PipelineResult.passed = no error finding in any report that ran; .per_validator = {name: caught / seconds / decided /
kinds / cached}; .validation_seconds = the sum of the validators' own seconds (overhead, also when cached).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .changeaware.validator import ChangeAwareValidator
from .compiler_checks.validator import CompilerChecksValidator
from .fixed_tests.validator import FixedTestsValidator
from .interface import Candidate, Finding, ValidationContext, ValidationReport
from .volta.validator import VoltaValidator

CONFIGS = {
    "A0": ("fixed_tests",),
    "A": ("compiler_checks", "fixed_tests"),
    "B": ("compiler_checks", "fixed_tests", "volta"),
    "C": ("compiler_checks", "fixed_tests", "changeaware", "volta"),
}
VALIDATORS = {"compiler_checks": CompilerChecksValidator, "fixed_tests": FixedTestsValidator,
              "changeaware": ChangeAwareValidator, "volta": VoltaValidator}
_CACHE_EXTRA = ("classifier", "budget", "gate_shapes", "gate_tools", "gate_parallel")


@dataclass
class PipelineResult:
    config: str
    reports: list[ValidationReport] = field(default_factory=list)
    seconds: float = 0.0                       # wall time of this call (small when reports came from the cache)
    stopped_after: str | None = None           # validator after which the pipeline stopped early
    skipped: list[str] = field(default_factory=list)   # validators of the configuration that did not run
    cached: list[str] = field(default_factory=list)    # validators whose report was reused from `cache`

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.reports for f in r.findings]

    @property
    def decided_by(self) -> dict[str, bool]:
        return {r.validator: r.decided for r in self.reports}

    @property
    def validation_seconds(self) -> float:
        return sum(r.seconds for r in self.reports)

    @property
    def per_validator(self) -> dict[str, dict]:
        return {r.validator: {"caught": not r.passed, "seconds": round(r.seconds, 3), "decided": r.decided,
                              "kinds": sorted({f.kind for f in r.findings if f.severity == "error"}),
                              "cached": r.validator in self.cached} for r in self.reports}

    def to_dict(self) -> dict:
        return {"config": self.config, "passed": self.passed, "seconds": self.seconds,
                "validation_seconds": self.validation_seconds, "stopped_after": self.stopped_after,
                "skipped": self.skipped, "cached": self.cached, "per_validator": self.per_validator,
                "reports": [r.to_dict() for r in self.reports]}


def _sha(text: str | None) -> str:
    return hashlib.sha1((text or "").encode()).hexdigest()[:16]


def _cache_key(cand: Candidate, name: str, ctx: ValidationContext) -> str:
    parts = {"v": name, "src": _sha(cand.source), "parent": _sha(cand.parent.source if cand.parent else None),
             "lang": cand.lang, "wl": cand.workload, "device": ctx.resolved_device(),
             "extra": {k: repr(ctx.extra.get(k)) for k in _CACHE_EXTRA}}
    if name == "volta":
        parts["shapes"] = ctx.volta_shapes
    if name in ("fixed_tests", "changeaware") and cand.lang == "tirx" and parts["device"] != "gpu":
        parts["numsim_racecheck"] = bool(ctx.extra.get("numsim_racecheck", True))     # only NumSim runs Racecheck
    return json.dumps(parts, sort_keys=True, default=str)


def run_pipeline(cand: Candidate, config: str = "C", ctx: ValidationContext | None = None,
                 stop_on_compile_error: bool = True, cache: dict | None = None, skip=()) -> PipelineResult:
    if config not in CONFIGS:
        raise ValueError(f"unknown configuration {config!r}; known: {', '.join(CONFIGS)}")
    ctx = (ctx or ValidationContext()).with_(config=config)
    names = CONFIGS[config]
    run_all = bool(ctx.extra.get("run_all"))
    gated = "compiler_checks" in names and "compiler_checks" not in skip
    tests_ctx = ctx.with_(extra={**ctx.extra, "numsim_racecheck": False}) if gated else ctx
    res, t0, suggested = PipelineResult(config), time.perf_counter(), None
    for i, name in enumerate(names):
        if name in skip:
            res.skipped.append(name)
            continue
        if name == "compiler_checks":
            vctx = ctx
        elif name == "fixed_tests":
            vctx = tests_ctx.with_(test_cases=None)
        elif name == "changeaware":
            vctx = tests_ctx
        else:
            vctx = ctx.with_(volta_shapes=suggested if config == "C" else None)
        key = _cache_key(cand, name, vctx) if cache is not None else None
        rep = cache.get(key) if cache is not None else None
        if rep is None:
            rep = VALIDATORS[name]().validate(cand, vctx)
            if cache is not None:
                cache[key] = rep
        else:
            res.cached.append(name)
        if name == "changeaware":
            suggested = rep.artifacts.get("suggested_volta_shapes")
        res.reports.append(rep)
        compile_err = any(f.kind == "compile_error" and f.severity == "error" for f in rep.findings)
        if (stop_on_compile_error and compile_err) or (not rep.passed and not run_all):
            res.stopped_after = name
            res.skipped += [n for n in names[i + 1:] if n not in res.skipped]
            break
    res.seconds = time.perf_counter() - t0
    return res
