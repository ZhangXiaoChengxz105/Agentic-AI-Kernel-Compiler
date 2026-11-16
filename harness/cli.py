"""Harness command line (needs a GPU node; see scripts/slurm/).

  python -m harness.cli check --op softmax [--variants softmax_opt,softmax_bug_tail] [--tests fixed|file.json]
      compile every selected manifest variant (verify.ptx mode=gpu), run the tests in an isolated child
      process per variant, print a summary, save results/harness/check_<op>_<ts>.json
  python -m harness.cli bench --op gemm [--shape "M=1024,N=1024,K=1024"] [--variants ...]
      correctness check on the bench shape first, then time every passing variant and the PyTorch
      implementation; save results/harness/bench_<op>_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from verify.cases import Variant, load_manifest  # noqa: E402
from verify.ptx import compile_to_ptx  # noqa: E402

from .fixed_suite import FIXED  # noqa: E402
from .testcase import TestCase  # noqa: E402

BENCH_SHAPES = {"softmax": {"rows": 4096, "N": 1024}, "layernorm": {"rows": 4096, "N": 1024},
                "gemm": {"M": 1024, "N": 1024, "K": 1024}}


def _variants(op: str, only: str | None, extra: list[str] | None = None):
    """Manifest variants (optionally filtered) + extra kernels given as NAME=PATH[:BLOCK] (e.g. LLM output).
    NAME must be the kernel's extern "C" entry name; BLOCK defaults to the reference op's candidate block."""
    _, variants = load_manifest(op)
    if only:
        keep = set(only.split(","))
        variants = [v for v in variants if v.name in keep]
    default_block = next((v.block for v in load_manifest(op)[1] if v.role == "candidate"), (128,))
    for spec in extra or []:
        name, rest = spec.split("=", 1)
        path, _, blk = rest.partition(":")
        block = tuple(int(b) for b in blk.split(",")) if blk else default_block
        variants.append(Variant(name, Path(path).resolve(), "candidate", block, None, None))
    return variants


def _compile(op: str, variants) -> dict:
    """label -> (ptx path, kernel entry name). For extra kernels the label is free-form: if it is not an
    entry in the PTX and the PTX has exactly one entry, that entry is used."""
    out = WORKSPACE / "build" / "ptx_gpu" / op
    res = {}
    for v in variants:      # extra files may share a stem (softmax_glm_r1.cu in several dirs) -> per-label dir
        b = compile_to_ptx(v.file, out / v.name, mode="gpu")
        if v.name in b.kernels:
            entry = v.name
        elif len(b.kernels) == 1:
            entry = b.kernels[0]
        else:
            raise SystemExit(f"{v.file}: cannot pick a kernel for label {v.name}; entries: {b.kernels}")
        res[v.name] = (b.ptx, entry)
    return res


def _save(kind: str, op: str, payload: dict) -> Path:
    d = WORKSPACE / "results" / "harness"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{kind}_{op}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def cmd_check(args) -> int:
    from .runner import run_tests_isolated
    variants = _variants(args.op, args.variants, args.extra)
    if args.tests in (None, "fixed"):
        tests = FIXED[args.op]
    else:
        tests = [TestCase.from_dict(t) for t in json.loads(Path(args.tests).read_text())]
    if args.repeat:
        tests = [TestCase(t.op, t.shape, t.dist, t.seed, args.repeat, t.tag) for t in tests]
    ptx = _compile(args.op, variants)
    all_rows, summary = [], []
    for v in variants:
        rows = run_tests_isolated(args.op, ptx[v.name][0], ptx[v.name][1], v.block, tests)
        n_pass = sum(r["passed"] for r in rows)
        reasons = sorted({r["reason"] for r in rows if not r["passed"]})
        summary.append((v.name, n_pass, len(rows), reasons))
        print(f"{v.name:26s} {n_pass:3d}/{len(rows):<3d} passed   {', '.join(reasons) or ''}")
        for r in rows:
            if not r["passed"]:
                print(f"    FAIL {TestCase.from_dict(r).key():52s} {r['reason']:16s} {r['message'][:80]}")
        all_rows += [{"variant": v.name, **r} for r in rows]
    p = _save("check", args.op, {"op": args.op, "tests": [t.to_dict() for t in tests], "results": all_rows})
    print(f"saved {p.relative_to(WORKSPACE)}")
    return 0


def cmd_bench(args) -> int:
    import torch

    from ref.ops import OPS
    from .bench import bench
    from .check import alloc_output
    from .launch import CudaModule
    from .runner import run_tests_isolated

    op = OPS[args.op]
    shape = ({k: int(v) for k, v in (kv.split("=") for kv in args.shape.split(","))} if args.shape
             else BENCH_SHAPES[args.op])
    variants = _variants(args.op, args.variants, args.extra)
    ptx = _compile(args.op, variants)
    tc = TestCase(args.op, shape, "normal", 0, 1, "bench")
    inputs = {k: v.cuda() for k, v in op.make_inputs(shape, "normal", 0).items()}
    numel = 1
    for s in op.out_shape(shape):
        numel *= s

    def metrics(ms):
        return {"ms": ms, "GB/s": op.bytes(shape) / ms / 1e6, "TFLOP/s": op.flops(shape) / ms / 1e9}

    rows = []
    t = bench(lambda: op.torch_impl(inputs), args.warmup, args.iters)
    rows.append({"impl": "torch", **t, **metrics(t["median_ms"])})
    for v in variants:
        ok = run_tests_isolated(args.op, ptx[v.name][0], ptx[v.name][1], v.block, [tc])[0]
        if not ok["passed"]:
            rows.append({"impl": v.name, "skipped": f"incorrect on bench shape: {ok['reason']}"})
            continue
        fn = CudaModule(ptx[v.name][0]).get(ptx[v.name][1])
        _, out = alloc_output(numel)
        grid, blk = op.launch(shape, v.block)
        a = op.kernel_args(inputs, out, shape)
        t = bench(lambda: fn(grid, blk, a), args.warmup, args.iters)
        rows.append({"impl": v.name, **t, **metrics(t["median_ms"])})
    base = rows[0]["ms"]
    print(f"op={args.op} shape={shape} (median of {args.iters}, L2 flushed)")
    for r in rows:
        if "skipped" in r:
            print(f"  {r['impl']:26s} skipped: {r['skipped']}")
        else:
            print(f"  {r['impl']:26s} {r['ms']:9.4f} ms  {r['GB/s']:8.1f} GB/s  {r['TFLOP/s']:7.3f} TFLOP/s"
                  f"  x{base / r['ms']:.2f} vs torch")
    p = _save("bench", args.op, {"op": args.op, "shape": shape, "gpu": torch.cuda.get_device_name(), "rows": rows})
    print(f"saved {p.relative_to(WORKSPACE)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--op", required=True, choices=sorted(FIXED))
    c.add_argument("--variants")
    c.add_argument("--tests", help="'fixed' (default) or a JSON file with a list of TestCase dicts")
    c.add_argument("--repeat", type=int, help="override repeat count for every test")
    c.add_argument("--extra", action="append", help="extra kernel NAME=PATH[:BLOCK] (repeatable)")
    b = sub.add_parser("bench")
    b.add_argument("--op", required=True, choices=sorted(FIXED))
    b.add_argument("--variants")
    b.add_argument("--shape")
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--iters", type=int, default=50)
    b.add_argument("--extra", action="append", help="extra kernel NAME=PATH[:BLOCK] (repeatable)")
    args = ap.parse_args()
    return cmd_check(args) if args.cmd == "check" else cmd_bench(args)


if __name__ == "__main__":
    sys.exit(main())
