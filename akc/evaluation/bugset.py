"""Configurations A0 / A / B / C on the injected bugs of every workload (seed of Task 5).

    akc eval-bugs [--workloads softmax,gemm,layernorm] [--lang cuda|tirx] [--device auto|gpu|cpu] [--configs A0,A,B,C]

For every bug variant, with the workload's correct version as the accepted parent (CUDA: <op>_opt; tirx: the bug-free
candidate, else the reference), each configuration runs through validation.pipeline.run_pipeline, exactly as the
agent loop would, with ctx.extra["run_all"] = True (a failure does not stop the later validators, so each one's catch
can be counted) and one report cache per bug (a validator with the same inputs in two configurations runs once):
    A0  fixed_tests
    A   compiler_checks + fixed_tests
    B   compiler_checks + fixed_tests + volta (default shape)
    C   compiler_checks + fixed_tests + changeaware + volta (change-aware shapes)
Per configuration: caught, validation seconds (sum of the validators' own seconds), and per validator caught / seconds /
decided / error kinds. On a CPU-only node the dynamic CUDA tests are reported "n/a" (not executed); tirx runs NumSim.
Output: table + results/eval/bugset_<lang>_<device>_<ts>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from ..paths import RESULTS
from ..task.workload import list_workloads, load_workload
from ..validation.interface import Candidate, ValidationContext
from ..validation.pipeline import CONFIGS, run_pipeline

COLUMNS = (("compiler_checks", "A", "gate"), ("fixed_tests", "A0", "fixed"), ("volta", "B", "VOLTA-B"),
           ("changeaware", "C", "CA tests"), ("volta", "C", "VOLTA-C"))


def _summ(r) -> dict:
    return {"passed": r.passed, "decided": r.decided, "seconds": round(r.seconds, 3), "errors": len(r.errors),
            "kinds": sorted({f.kind for f in r.findings if f.severity == "error"}),
            "artifacts": {k: v for k, v in r.artifacts.items()
                          if k in ("categories", "suggested_volta_shapes", "runs", "shapes", "stages", "sanitizer")}}


def _cell(s: dict | None) -> str:
    if s is None:
        return "-"
    if not s["decided"] and s["passed"]:
        return "n/a"
    return "caught " + ",".join(s["kinds"])[:14] if not s["passed"] else "missed"


def parent_variant(wl, lang: str):
    """The correct version the injected bugs were made from: the bug-free candidate (CUDA: <op>_opt) or, when a
    language has none (tirx softmax: the bugs are single edits of the reference), the reference itself."""
    return next((v for v in wl.candidates(lang) if not v.bug), None) or wl.reference(lang)


def evaluate(cand: Candidate, ctx: ValidationContext, configs=tuple(CONFIGS), skip=(), cache: dict | None = None) -> dict:
    """{config: {"caught", "seconds", "validators": {name: summary}, "stopped_after", "cached"}} for one candidate.
    `cache`: a fresh report cache for this candidate (default {}); evaluation.volta_pool passes one whose VOLTA
    reports come from parallel CPU workers."""
    ctx = ctx.with_(extra={**ctx.extra, "run_all": True})
    cache = {} if cache is None else cache
    out = {}
    for cfg in configs:
        res = run_pipeline(cand, cfg, ctx, cache=cache, skip=skip)
        out[cfg] = {"caught": not res.passed, "seconds": round(res.validation_seconds, 3),
                    "validators": {r.validator: _summ(r) for r in res.reports}, "per_validator": res.per_validator,
                    "stopped_after": res.stopped_after, "cached": res.cached, "skipped": res.skipped}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc eval-bugs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", default=",".join(list_workloads()))
    ap.add_argument("--lang", default="cuda", choices=["cuda", "tirx"])
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset of A0,A,B,C")
    a = ap.parse_args(argv)
    configs = [c for c in a.configs.split(",") if c]
    ctx = ValidationContext(device=a.device)
    device = ctx.resolved_device()
    rows = []
    d = RESULTS / "eval"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"bugset_{a.lang}_{device}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"

    def save(done: bool) -> tuple[dict, dict]:
        tot = {c: sum(r["detected"][c] for r in rows) for c in configs}
        secs = {c: round(sum(r["configs"][c]["seconds"] for r in rows), 1) for c in configs}
        f.write_text(json.dumps({"device": device, "lang": a.lang, "configs": configs,
                                 "config_validators": {c: list(CONFIGS[c]) for c in configs}, "totals": tot,
                                 "seconds": secs, "n": len(rows), "complete": done, "rows": rows}, indent=2, default=str))
        return tot, secs
    print(f"device={device} lang={a.lang} configs={','.join(configs)}")
    hdr = (f"{'bug':26s} {'truth':9s} {'predicted':20s} " + " ".join(f"{h:21s}" for _, _, h in COLUMNS) + "  "
           + " ".join(f"{c:>2s}" for c in configs))
    print(hdr + "\n" + "-" * len(hdr))
    for op in a.workloads.split(","):
        wl = load_workload(op)
        bugs = wl.bugs(a.lang)
        if not bugs:
            continue
        good = parent_variant(wl, a.lang)
        parent = Candidate.from_file(op, good.file, entry=good.name if a.lang == "cuda" else None, label=good.name)
        for b in bugs:
            cand = Candidate.from_file(op, b.file, entry=b.name if a.lang == "cuda" else None, block=b.block,
                                       parent=parent, label=b.name)
            res = evaluate(cand, ctx, configs)
            cells = [_cell(res.get(cfg, {}).get("validators", {}).get(v)) for v, cfg, _ in COLUMNS]
            ca = res.get("C", {}).get("validators", {}).get("changeaware") or {}
            print(f"{b.name:26s} {b.bug['category']:9s} {','.join(ca.get('artifacts', {}).get('categories', []))[:20]:20s} "
                  + " ".join(f"{c:21s}" for c in cells) + "  "
                  + " ".join(f"{'x' if res[c]['caught'] else '.':>2s}" for c in configs))
            rows.append({"workload": op, "bug": b.name, "truth": b.bug["category"], "parent": good.name,
                         "configs": res, "detected": {c: res[c]["caught"] for c in configs}})
            save(False)                                    # partial results survive a Slurm time limit
    n = len(rows)
    tot, secs = save(True)
    print("\ndetected  " + "   ".join(f"{c}: {tot[c]}/{n}" for c in configs)
          + "\nvalidation seconds (sum over bugs)  " + "   ".join(f"{c}: {secs[c]}" for c in configs)
          + ("   (dynamic tests not executed on this device: the test parts are lower bounds)"
             if device != "gpu" and a.lang == "cuda" else ""))
    print(f"saved {f}")
    return 0
