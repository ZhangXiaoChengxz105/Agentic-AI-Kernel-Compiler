"""akc command line.  Usage: akc <command> [options]   (or: python -m akc <command> ...)

  validate   run configuration A0/A/B/C on one kernel file and print the feedback the agent would get
             (--perf: if it passes, also measure its speed on the GPU and add the speed feedback)
  hottest    change-aware check of a kernel revision: what changed, targeted tests, problems found
  check      run tests (fixed suite or a JSON list) on workload variants and/or extra kernel files
             (GPU; --lang tirx: the tirx-lite variants, also on the CPU through NumSim)
  bench      speed of kernels on the GPU: time, x PyTorch, roofline (% of peak), practical floor
  volta      VOLTA on every candidate variant of a workload (--lang cuda|tirx), checked against manifest
             expectations (CPU)
  eval-bugs  configurations A0/A/B/C on the injected bugs (detection, coverage, overhead per validator)
  mutants    mutation benchmark: is each generated edit classified right (CPU), caught by A / C (GPU)?
             (--classifier ir|ast: tirx-lite rows use the IR change analysis or the Python-AST rules)
  tirx-gpu   tirx-lite kernels on the GPU (correctness + speed)
  agent      the agent repair loop (Task 4): model -> configuration A0/A/B/C -> feedback, <= 10 iterations
             (spends Lumen coins; --dry-run sends nothing)
  pilot      GLM pilot loop (spends Lumen coins)
  lumen      Lumen client: models | usage | chat
Run `akc <command> -h` for the options of a command.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _candidates(workload: str, variants: str | None, extra: list[str] | None, lang: str = "cuda"):
    from .task.workload import load_workload
    from .validation.interface import Candidate
    wl = load_workload(workload)
    out = []
    for v in wl.variants:
        if v.lang == lang and (variants is None or v.name in variants.split(",")):
            out.append(Candidate.from_file(workload, v.file, entry=v.name if lang == "cuda" else None, block=v.block, label=v.name))
    for spec in extra or []:                                  # LABEL=PATH[:BLOCK]
        label, rest = spec.split("=", 1)
        path, _, blk = rest.partition(":")
        block = tuple(int(b) for b in blk.split(",")) if blk else None
        out.append(Candidate.from_file(workload, path, block=block, label=label))
    return out


def cmd_validate(argv):
    ap = argparse.ArgumentParser(prog="akc validate")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--parent", help="last accepted version (default: workload reference)")
    ap.add_argument("--config", default="C", choices=["A0", "A", "B", "C"],
                    help="A0 fixed tests; A + compiler-checks gate; B + VOLTA; C + change-aware tests")
    ap.add_argument("--run-all", action="store_true",
                    help="run every validator of the configuration even after a failure (default: stop at the first)")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--perf", action="store_true", help="if validation passes, measure speed too (GPU)")
    ap.add_argument("--llm-classify", action="store_true",
                    help="configuration C: also ask an LLM for change categories and risky sizes (spends Lumen coins)")
    ap.add_argument("--llm-model", default="glm-5.3")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from .feedback.render import render_observation
    from .perf import measure
    from .validation import Candidate, ValidationContext, run_pipeline
    parent = Candidate.from_file(a.workload, a.parent, label="parent") if a.parent else None
    cand = Candidate.from_file(a.workload, a.kernel, parent=parent)
    extra = {"run_all": True} if a.run_all else {}
    if a.llm_classify:
        from .validation.changeaware.llm_classify import LlmClassifier
        extra["llm_classifier"] = LlmClassifier(model=a.llm_model)
    res = run_pipeline(cand, a.config, ValidationContext(device=a.device, extra=extra))
    for r in res.reports:
        print(f"{r.validator:12s} {'PASS' if r.passed else 'FAIL'} decided={r.decided} errors={len(r.errors)} {r.seconds:.1f}s")
    perf = measure(cand, device=a.device) if a.perf and res.passed else None   # speed only for passing kernels
    if perf is not None:
        print(f"{'perf':12s} {'measured' if perf.measured else 'not measured'} {perf.seconds:.1f}s")
    print("\n" + render_observation(res.reports, perf))
    if a.json:
        out = res.to_dict()
        if perf is not None:
            out["perf"] = perf.to_dict()
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0 if res.passed else 1


def cmd_check(argv):
    ap = argparse.ArgumentParser(prog="akc check")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--lang", default="cuda", choices=["cuda", "tirx"], help="which manifest variants (default cuda)")
    ap.add_argument("--variants", help="comma-separated manifest variants (default: all variants of --lang)")
    ap.add_argument("--extra", action="append", help="extra kernel LABEL=PATH[:BLOCK] (repeatable)")
    ap.add_argument("--tests", help="'fixed' (default) or a JSON file with a list of TestCase dicts")
    ap.add_argument("--repeat", type=int)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = ap.parse_args(argv)
    from .paths import RESULTS
    from .task.testcase import TestCase
    from .validation.fixed_tests.suite import fixed_suite
    from .validation.runners import run
    tests = fixed_suite(a.workload) if a.tests in (None, "fixed") else \
        [TestCase.from_dict(t) for t in json.loads(Path(a.tests).read_text())]
    if a.repeat:
        tests = [TestCase(t.op, t.shape, t.dist, t.seed, a.repeat, t.tag, t.misalign) for t in tests]
    rows = []
    for c in _candidates(a.workload, a.variants, a.extra, a.lang):
        outs = run(c, tests, a.device)
        reasons = sorted({o["reason"] for o in outs if not o["passed"]})
        print(f"{c.label:26s} {sum(o['passed'] for o in outs):3d}/{len(outs):<3d} passed   {', '.join(reasons)}")
        for o in outs:
            if not o["passed"]:
                print(f"    FAIL {TestCase.from_dict(o).key():60s} {o['reason']:16s} {o['message'][:80]}")
        rows += [{"variant": c.label, **o} for o in outs]
    d = RESULTS / "harness"
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if a.lang == "cuda" else f"_{a.lang}"
    f = d / f"check_{a.workload}{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"workload": a.workload, "lang": a.lang, "tests": [t.to_dict() for t in tests], "results": rows}, indent=2, default=str))
    print(f"saved {f}")
    return 0


def cmd_bench(argv):
    ap = argparse.ArgumentParser(prog="akc bench", description="Speed on the GPU (perf.measure): correctness at the "
                                 "benchmark shape, then time vs PyTorch, roofline numbers and the practical floor.")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--lang", default="cuda", choices=["cuda", "tirx"], help="which manifest variants (default cuda)")
    ap.add_argument("--variants", help="comma-separated manifest variants (default: the bug-free ones)")
    ap.add_argument("--extra", action="append", help="extra kernel LABEL=PATH[:BLOCK] (repeatable)")
    ap.add_argument("--shape", help='e.g. "rows=4096,N=1024" (default: manifest perf.shape)')
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-check", action="store_true", help="skip the correctness check at the benchmark shape")
    ap.add_argument("--feedback", action="store_true", help="also print the speed feedback text the agent would get")
    a = ap.parse_args(argv)
    from .feedback.render import render_perf
    from .paths import RESULTS
    from .perf import measure
    from .task.workload import load_workload
    shape = {k: int(v) for k, v in (kv.split("=") for kv in a.shape.split(","))} if a.shape else None
    variants = a.variants or ",".join(v.name for v in load_workload(a.workload).variants if v.lang == a.lang and not v.bug)
    reports = [measure(c, shape, iters=a.iters, warmup=a.warmup, check=not a.no_check)
               for c in _candidates(a.workload, variants, a.extra, a.lang)]
    done = [r for r in reports if r.measured]
    print(f"workload={a.workload} shape={reports[0].shape if reports else shape} gpu={done[0].gpu if done else '-'} "
          f"(median of {a.iters}, L2 flushed before every launch)")
    if done:
        r0 = done[0]
        copy = f", device copy of the same bytes {r0.copy_ms:.4f} ms" if r0.copy_ms else ""
        print(f"  PyTorch {r0.torch_ms:.4f} ms{copy}; {r0.bound}-bound ({r0.intensity:.2f} FLOP/byte)")
    print(f"  {'kernel':24s} {'ms':>9s} {'xTorch':>7s} {'GB/s':>7s} {'%bw':>5s} {'TFLOP/s':>8s} {'%fp32':>6s} {'floor':>6s}")
    for r in reports:
        if not r.measured:
            print(f"  {r.label:24s} not measured: {r.reason[:150]}")
            continue
        print(f"  {r.label:24s} {r.time_ms:9.4f} {r.speedup_vs_torch:6.2f}x {r.achieved_gbps:7.0f} {r.pct_peak_bw:4.0f}% "
              f"{r.achieved_tflops:8.3f} {r.pct_peak_flops:5.0f}% {r.headroom:5.2f}x")
    if a.feedback:
        for r in reports:
            print(f"\n--- feedback for {r.label}\n{render_perf(r)}")
    if not done:
        print("nothing measured: no result file written")
        return 1
    d = RESULTS / "perf"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"bench_{a.workload}_{a.lang}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"workload": a.workload, "reports": [r.to_dict() for r in reports]}, indent=1, default=str))
    print(f"saved {f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "validate":
        return cmd_validate(rest)
    if cmd == "check":
        return cmd_check(rest)
    if cmd == "bench":
        return cmd_bench(rest)
    if cmd == "hottest":
        from .validation.changeaware.hottest import main as m
    elif cmd == "volta":
        from .evaluation.volta_matrix import main as m
    elif cmd == "eval-bugs":
        from .evaluation.bugset import main as m
    elif cmd == "mutants":
        from .evaluation.mutants import main as m
    elif cmd == "tirx-gpu":
        from .evaluation.gpu_tirx import main as m
    elif cmd == "agent":
        from .agent.loop import main as m
    elif cmd == "pilot":
        from .agent.pilot import main as m
    elif cmd == "lumen":
        from .llm import lumen
        sys.argv = ["akc lumen", *rest]
        return lumen.main()
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    return m(rest)


if __name__ == "__main__":
    sys.exit(main())
