"""hottest: offline one-shot check of a kernel revision.  Input a kernel (+ previous accepted version),
get back what changed, which risks that implies, the targeted tests/VOLTA shapes, and the problems found.

    akc hottest --workload softmax --new my_kernel.cu [--old previous.cu] [--device auto|gpu|cpu]
                [--no-volta] [--with-fixed] [--budget 16] [--json report.json]

  --old omitted -> compared with the workload's reference kernel in the same language
  device cpu    -> CUDA tests cannot run (reported as "not verified"); VOLTA still runs (CPU);
                   tirx-lite candidates run on TIRx NumSim instead of the GPU
Exit code: 0 no errors, 1 errors found, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...feedback.render import render
from ...task.workload import list_workloads
from ..fixed_tests.validator import FixedTestsValidator
from ..interface import Candidate, ValidationContext
from ..volta.validator import VoltaValidator
from .validator import ChangeAwareValidator


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc hottest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=list_workloads())
    ap.add_argument("--new", required=True, help="kernel file (.cu or tirx-lite .py)")
    ap.add_argument("--old", help="previous ACCEPTED version (default: the workload reference)")
    ap.add_argument("--entry", help="CUDA kernel entry name (default: the only entry in the file)")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--no-volta", action="store_true")
    ap.add_argument("--with-fixed", action="store_true", help="also run the fixed suite (configuration A's tests)")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--json", help="write the full reports here")
    ap.add_argument("--llm-classify", action="store_true", help="also ask an LLM (spends Lumen coins)")
    ap.add_argument("--llm-model", default="glm-5.3")
    a = ap.parse_args(argv)

    parent = Candidate.from_file(a.workload, a.old, label="old") if a.old else None
    cand = Candidate.from_file(a.workload, a.new, entry=a.entry, parent=parent, label=Path(a.new).stem)
    extra = {"budget": a.budget}
    if a.llm_classify:
        from .llm_classify import LlmClassifier
        extra["llm_classifier"] = LlmClassifier(model=a.llm_model)
    ctx = ValidationContext(config="C", device=a.device, extra=extra)
    reports = []
    ca = ChangeAwareValidator().validate(cand, ctx)
    reports.append(ca)
    if not a.no_volta and cand.lang == "cuda":
        reports.append(VoltaValidator().validate(cand, ctx.with_(volta_shapes=ca.artifacts["suggested_volta_shapes"])))
    if a.with_fixed:
        reports.append(FixedTestsValidator().validate(cand, ctx))

    art = ca.artifacts
    print(f"== hottest: {a.new}  (workload {a.workload}, {cand.lang}, compared with {art['parent']}, device {ctx.resolved_device()})")
    print(f"change categories: {', '.join(art['categories']) or 'none'}   tiles seen in kernel: {art['tiles'] or '-'}")
    for lab in art["labels"][:8]:
        print(f"   {lab['category']:9s} {lab['rule']:16s} L{lab['line']}: {lab['evidence'][:90]}")
    if art.get("llm"):
        li = art["llm"]
        print(f"LLM ({li['model']}): {li['n_labels']} labels, risky sizes {li['shapes'] or '-'}, "
              f"{li['cost']:.4f} coins, {li['latency_s']:.1f}s{' (cached)' if li['cached'] else ''}"
              f"{'  ERROR ' + li['error'] if li['error'] else ''}")
    ran = sum(o.get("reason") != "skipped" for o in art["outcomes"])
    print(f"targeted tests: {len(art['tests'])} generated, {ran} executed   VOLTA shapes: {art['suggested_volta_shapes'] or '-'}")
    for r in reports:
        state = "PASS" if r.passed else "FAIL"
        print(f"   {r.validator:12s} {state}  decided={r.decided}  errors={len(r.errors)}  {r.seconds:.1f}s")
    print("\n" + render(reports))
    if a.json:
        Path(a.json).write_text(json.dumps([r.to_dict() for r in reports], indent=1, default=str))
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
