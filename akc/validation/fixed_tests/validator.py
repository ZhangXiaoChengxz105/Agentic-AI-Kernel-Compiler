"""FixedTestsValidator (baseline, configuration A).

Input : Candidate; ctx.test_cases (None -> the workload's fixed suite); ctx.device
Output: ValidationReport("fixed_tests"): one finding per failing test (mismatch, nonfinite, out_of_bounds,
        unwritten, nondeterministic, runtime/compile error); tests that could not run -> info "skipped".
        decided = at least one test actually executed.
"""
from __future__ import annotations

import time

from ..interface import Candidate, ValidationContext, ValidationReport
from ..runners import outcome_to_finding, run
from .suite import fixed_suite


class FixedTestsValidator:
    name = "fixed_tests"

    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport:
        t0 = time.perf_counter()
        tests = ctx.test_cases if ctx.test_cases is not None else fixed_suite(cand.workload)
        outcomes = run(cand, tests, ctx.resolved_device(), ctx.extra.get("numsim_racecheck", True))
        findings = [f for f in (outcome_to_finding(o, self.name) for o in outcomes) if f]
        executed = sum(o.get("reason") != "skipped" for o in outcomes)
        return ValidationReport.build(self.name, findings, executed > 0, t0, outcomes=outcomes, n_tests=len(tests))
