"""ChangeAwareValidator (validation mechanism 2, Task 2).

Input : Candidate with `parent` = last ACCEPTED version (None -> the workload reference in the same
        language; if the workload has no reference in that language, every category is assumed);
        ctx.device; ctx.extra["budget"] (max targeted tests, default 16; floor tests come on top);
        ctx.extra["llm_classifier"] (optional LlmClassifier: extra categories + risky sizes, spends coins);
        ctx.extra["classifier"] ("auto" = IR analysis for tirx-lite, "ast" = Python-AST rules)
Steps : diff(parent, candidate) -> classify -> generate targeted tests + VOLTA shapes -> run the tests with
        the shared executor (GPU, or NumSim for tirx on CPU) -> findings
Output: ValidationReport("changeaware")
  findings     one per failing test; evidence includes the test's purpose (tag) and the related change
               lines (labels of the same category), so feedback can say "your edit on line 20 broke N=129"
  artifacts    categories, labels, tests, suggested_volta_shapes (the pipeline hands these to VOLTA),
               tiles found in the kernel, outcomes, analysis (mode ir / ast-fallback / cuda-text, traced shapes,
               hunk counts, fallback reason, ir_blind)
  decided      at least one generated test executed
"""
from __future__ import annotations

import time
from dataclasses import asdict

from ...task.workload import load_workload
from ..interface import Candidate, Finding, ValidationContext, ValidationReport
from ..runners import outcome_to_finding, run
from . import plan


def parent_source(cand: Candidate) -> tuple[str | None, str]:
    if cand.parent is not None:
        return cand.parent.source, cand.parent.label or "parent"
    wl = load_workload(cand.workload)
    ref = wl.meta.get("references", {}).get(cand.lang)
    if ref is None:
        return None, "none"
    v = wl.variant(ref)
    return v.file.read_text(), f"reference {v.name}"


class ChangeAwareValidator:
    name = "changeaware"

    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport:
        t0 = time.perf_counter()
        old, parent_label = parent_source(cand)
        p = plan(cand.workload, old, cand.source, cand.lang, ctx.extra.get("budget", 16),
                 llm=ctx.extra.get("llm_classifier"), classifier=ctx.extra.get("classifier", "auto"))
        outcomes = run(cand, p.tests, ctx.resolved_device(), ctx.extra.get("numsim_racecheck", True))
        findings = []
        for o in outcomes:
            f = outcome_to_finding(o, self.name)
            if f is None:
                continue
            cat = (o.get("tag") or "").split(":")[0]
            related = list(dict.fromkeys(f"L{lab.line}: {lab.evidence[:120]}"     # dedupe: several rules can
                                         for lab in p.labels if lab.category == cat))[:3]  # label one line
            if related:
                f.evidence["related_change"] = related
            findings.append(f)
        executed = sum(o.get("reason") != "skipped" for o in outcomes)
        if not p.tests:
            findings.append(Finding("skipped", "info", self.name, "no tests generated"))
        return ValidationReport.build(
            self.name, findings, executed > 0, t0, parent=parent_label, categories=p.cats,
            labels=[asdict(lab) for lab in p.labels], tiles=p.tiles, tests=[t.to_dict() for t in p.tests],
            suggested_volta_shapes=p.volta, outcomes=outcomes, llm=p.llm, analysis=p.analysis)
