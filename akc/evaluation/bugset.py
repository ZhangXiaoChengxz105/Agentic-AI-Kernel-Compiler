"""Configurations A / B / C on the injected bugs of every workload (seed of Task 5).

    akc eval-bugs [--workloads softmax,gemm,layernorm] [--lang cuda|tirx] [--device auto|gpu|cpu]

For every bug variant, with the workload's correct `<op>_opt` as the accepted parent:
    fixed        FixedTestsValidator                       (needs a GPU for CUDA)
    volta_B      VoltaValidator at the workload default shape
    changeaware  ChangeAwareValidator (targeted tests)      (needs a GPU for CUDA; tirx can use NumSim)
    volta_C      VoltaValidator at the change-aware shapes
    detected     A = fixed;  B = fixed or volta_B;  C = fixed or changeaware or volta_C
Each component keeps `decided` (verification coverage) and `seconds` (validation overhead).
On a CPU-only node the dynamic tests are reported "n/a" (not executed), VOLTA columns still run.
Output: table + results/eval/bugset_<ts>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from ..paths import RESULTS
from ..task.workload import list_workloads, load_workload
from ..validation.changeaware.validator import ChangeAwareValidator
from ..validation.fixed_tests.validator import FixedTestsValidator
from ..validation.interface import Candidate, ValidationContext
from ..validation.volta.validator import VoltaValidator


def _summ(r) -> dict:
    return {"passed": r.passed, "decided": r.decided, "seconds": round(r.seconds, 3), "errors": len(r.errors),
            "kinds": sorted({f.kind for f in r.findings if f.severity == "error"}),
            "artifacts": {k: v for k, v in r.artifacts.items() if k in ("categories", "suggested_volta_shapes", "runs", "shapes")}}


def _cell(s: dict) -> str:
    if not s["decided"] and s["passed"]:
        return "n/a"
    return "caught " + ",".join(s["kinds"])[:18] if not s["passed"] else "missed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc eval-bugs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", default=",".join(list_workloads()))
    ap.add_argument("--lang", default="cuda", choices=["cuda", "tirx"])
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    a = ap.parse_args(argv)
    ctx = ValidationContext(device=a.device)
    device = ctx.resolved_device()
    rows = []
    print(f"device={device} lang={a.lang}")
    hdr = f"{'bug':26s} {'truth':9s} {'predicted':20s} {'fixed':12s} {'VOLTA-B':16s} {'CA tests':16s} {'VOLTA-C':16s}  A B C"
    print(hdr + "\n" + "-" * len(hdr))
    for op in a.workloads.split(","):
        wl = load_workload(op)
        bugs = wl.bugs(a.lang)
        if not bugs:
            continue
        good = next(v for v in wl.candidates(a.lang) if not v.bug)
        parent = Candidate.from_file(op, good.file, entry=good.name if a.lang == "cuda" else None, label=good.name)
        for b in bugs:
            cand = Candidate.from_file(op, b.file, entry=b.name if a.lang == "cuda" else None, block=b.block,
                                       parent=parent, label=b.name)
            fx = _summ(FixedTestsValidator().validate(cand, ctx))
            ca_rep = ChangeAwareValidator().validate(cand, ctx)
            ca = _summ(ca_rep)
            vb = _summ(VoltaValidator().validate(cand, ctx))
            vc = _summ(VoltaValidator().validate(cand, ctx.with_(volta_shapes=ca_rep.artifacts["suggested_volta_shapes"] or None)))
            det = {"A": not fx["passed"], "B": not fx["passed"] or not vb["passed"],
                   "C": not fx["passed"] or not ca["passed"] or not vc["passed"]}
            print(f"{b.name:26s} {b.bug['category']:9s} {','.join(ca['artifacts'].get('categories', []))[:20]:20s} "
                  f"{_cell(fx):12s} {_cell(vb):16s} {_cell(ca):16s} {_cell(vc):16s}  "
                  + " ".join("x" if det[k] else "." for k in "ABC"))
            rows.append({"workload": op, "bug": b.name, "truth": b.bug["category"], "fixed": fx, "volta_B": vb,
                         "changeaware": ca, "volta_C": vc, "detected": det})
    n = len(rows)
    tot = {k: sum(r["detected"][k] for r in rows) for k in "ABC"}
    print(f"\ndetected  A: {tot['A']}/{n}   B: {tot['B']}/{n}   C: {tot['C']}/{n}"
          + ("   (dynamic tests not executed on this device: A and the test part of C are lower bounds)" if device != "gpu" and a.lang == "cuda" else ""))
    d = RESULTS / "eval"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"bugset_{a.lang}_{device}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"device": device, "lang": a.lang, "totals": tot, "n": n, "rows": rows}, indent=2, default=str))
    print(f"saved {f}")
    return 0
