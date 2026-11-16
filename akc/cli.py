"""akc command line.  Usage: akc <command> [options]   (or: python -m akc <command> ...)

  validate   run configuration A/B/C on one kernel file and print the feedback the agent would get
  hottest    change-aware check of a kernel revision: what changed, targeted tests, problems found
  check      run tests (fixed suite or a JSON list) on workload variants and/or extra kernel files (GPU)
  bench      time workload variants / extra kernels against PyTorch (GPU)
  volta      VOLTA on every CUDA variant of a workload, checked against manifest expectations (CPU)
  eval-bugs  configurations A/B/C on the injected bugs (detection, coverage, overhead)
  tirx-gpu   tirx-lite kernels on the GPU (correctness + speed)
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
    ap.add_argument("--config", default="C", choices=["A", "B", "C"])
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from .feedback.render import render
    from .validation import Candidate, ValidationContext, run_pipeline
    parent = Candidate.from_file(a.workload, a.parent, label="parent") if a.parent else None
    res = run_pipeline(Candidate.from_file(a.workload, a.kernel, parent=parent), a.config, ValidationContext(device=a.device))
    for r in res.reports:
        print(f"{r.validator:12s} {'PASS' if r.passed else 'FAIL'} decided={r.decided} errors={len(r.errors)} {r.seconds:.1f}s")
    print("\n" + render(res.reports))
    if a.json:
        Path(a.json).write_text(json.dumps(res.to_dict(), indent=1, default=str))
    return 0 if res.passed else 1


def cmd_check(argv):
    ap = argparse.ArgumentParser(prog="akc check")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--variants", help="comma-separated manifest variants (default: all CUDA variants)")
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
    for c in _candidates(a.workload, a.variants, a.extra):
        outs = run(c, tests, a.device)
        reasons = sorted({o["reason"] for o in outs if not o["passed"]})
        print(f"{c.label:26s} {sum(o['passed'] for o in outs):3d}/{len(outs):<3d} passed   {', '.join(reasons)}")
        for o in outs:
            if not o["passed"]:
                print(f"    FAIL {TestCase.from_dict(o).key():60s} {o['reason']:16s} {o['message'][:80]}")
        rows += [{"variant": c.label, **o} for o in outs]
    d = RESULTS / "harness"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"check_{a.workload}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"workload": a.workload, "tests": [t.to_dict() for t in tests], "results": rows}, indent=2, default=str))
    print(f"saved {f}")
    return 0


def cmd_bench(argv):
    ap = argparse.ArgumentParser(prog="akc bench")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--variants")
    ap.add_argument("--extra", action="append")
    ap.add_argument("--shape", help='e.g. "rows=4096,N=1024" (default per workload)')
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args(argv)
    import torch

    from .backend.bench import bench
    from .backend.launch import CudaModule
    from .backend.ptx import compile_to_ptx
    from .paths import BUILD, RESULTS
    from .task.reference import OPS
    from .task.testcase import TestCase
    from .task.workload import load_workload
    from .validation.runners import run
    from .validation.runners.check import alloc_output
    defaults = {"softmax": {"rows": 4096, "N": 1024}, "layernorm": {"rows": 4096, "N": 1024}, "gemm": {"M": 1024, "N": 1024, "K": 1024}}
    shape = {k: int(v) for k, v in (kv.split("=") for kv in a.shape.split(","))} if a.shape else defaults[a.workload]
    op, wl = OPS[a.workload], load_workload(a.workload)
    inputs = {k: v.cuda() for k, v in op.make_inputs(shape, "normal", 0).items()}
    numel = int(torch.tensor(op.out_shape(shape)).prod())
    met = lambda ms: {"ms": ms, "GB/s": op.bytes(shape) / ms / 1e6, "TFLOP/s": op.flops(shape) / ms / 1e9}  # noqa: E731
    t = bench(lambda: op.torch_impl(inputs), a.warmup, a.iters)
    rows = [{"impl": "torch", **t, **met(t["median_ms"])}]
    for c in _candidates(a.workload, a.variants, a.extra):
        ok = run(c, [TestCase(a.workload, shape, "normal", 0, 1, "bench")], "gpu")[0]
        if not ok["passed"]:
            rows.append({"impl": c.label, "skipped": f"incorrect on bench shape: {ok['reason']}"})
            continue
        b = compile_to_ptx(c.materialize(), BUILD / "ptx_gpu" / c.label, mode="gpu")
        fn = CudaModule(b.ptx).get(c.entry or b.kernels[0])
        _, out = alloc_output(numel)
        grid, blk = op.launch(shape, c.block or tuple(wl.launch["block"]))
        args = op.kernel_args(inputs, out, shape)
        t = bench(lambda: fn(grid, blk, args), a.warmup, a.iters)
        rows.append({"impl": c.label, **t, **met(t["median_ms"])})
    base = rows[0]["ms"]
    print(f"workload={a.workload} shape={shape} (median of {a.iters}, L2 flushed)")
    for r in rows:
        print(f"  {r['impl']:26s} " + (f"skipped: {r['skipped']}" if "skipped" in r else
              f"{r['ms']:9.4f} ms  {r['GB/s']:8.1f} GB/s  {r['TFLOP/s']:7.3f} TFLOP/s  x{base / r['ms']:.2f} vs torch"))
    d = RESULTS / "harness"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"bench_{a.workload}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"workload": a.workload, "shape": shape, "gpu": torch.cuda.get_device_name(), "rows": rows}, indent=2))
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
    elif cmd == "tirx-gpu":
        from .evaluation.gpu_tirx import main as m
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
