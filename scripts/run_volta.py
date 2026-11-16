#!/usr/bin/env python3
"""Building block 4: end-to-end VOLTA loop for one operator (softmax | gemm | layernorm).

  every variant in kernels/<op>/manifest.json : .cu --(verify.ptx, mode=volta)--> .ptx
  every shape x every candidate               : volta compare(ref, cand) --> VoltaResult
  compare with the manifest's expect_volta, print a table, save JSON.

Usage:
  source env.sh
  envs/tirx/bin/python scripts/run_volta.py --op gemm                       # shapes from manifest
  envs/tirx/bin/python scripts/run_volta.py --op softmax --shapes "N=1024;N=1000" --timeout 300
Output: stdout table; results/volta/<op>_<timestamp>.json; PTX under build/ptx/<op>/.
Exit code 0 iff every result matched expectations. CPU only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import WORKSPACE  # noqa: E402
from verify.cases import LAYOUTS, expected_statuses, load_manifest, make_case, shape_label  # noqa: E402
from verify.ptx import compile_to_ptx  # noqa: E402
from verify.volta import run_compare  # noqa: E402


def parse_shapes(text: str) -> list[dict]:
    shapes = []
    for chunk in text.split(";"):
        if chunk.strip():
            shapes.append({k.strip(): int(v) for k, v in (kv.split("=") for kv in chunk.split(","))})
    return shapes


def tool_versions() -> dict:
    def run(cmd, cwd=None):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()
        except OSError as e:
            return f"unavailable: {e}"
    nvcc = run(["nvcc", "--version"]).splitlines()
    return {"nvcc": nvcc[-1] if nvcc else "?",
            "volta_commit": run(["git", "rev-parse", "--short", "HEAD"], cwd=WORKSPACE / "third_party" / "volta")}


def describe(r) -> str:
    if r.status == "EQUIV":
        return f"{r.elements_checked} elements proved equal"
    if r.status == "NOT_EQUIV":
        return f"{r.mismatches} elements differ, e.g. {', '.join(r.mismatch_examples[:3])}"
    return (f"[{r.which}] " if r.which else "") + r.detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--shapes", help='e.g. "N=256;N=200" or "M=32,N=32,K=40" (default: manifest volta_shapes)')
    ap.add_argument("--only", help="comma-separated candidate names to run (default: all)")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per VOLTA run")
    ap.add_argument("--out-dir", default=str(WORKSPACE / "results" / "volta"))
    args = ap.parse_args()

    meta, variants = load_manifest(args.op)
    shapes = parse_shapes(args.shapes) if args.shapes else meta["volta_shapes"]
    ref = next(v for v in variants if v.role == "reference")
    cands = [v for v in variants if v.role == "candidate"]
    if args.only:
        keep = set(args.only.split(","))
        cands = [c for c in cands if c.name in keep]

    build_dir = WORKSPACE / "build" / "ptx" / args.op
    builds, compile_log = {}, []
    for v in [ref, *cands]:
        b = compile_to_ptx(v.file, build_dir, mode="volta")
        if v.name not in b.kernels:
            raise SystemExit(f"{v.file}: kernel {v.name} not in PTX entries {b.kernels}")
        builds[v.name] = b
        compile_log.append({"variant": v.name, "ptx": str(b.ptx), "seconds": round(b.seconds, 3),
                            "cmd": list(b.cmd), "warnings": b.warnings})
    print(f"[compile] {len(builds)} variants of {args.op} -> {build_dir.relative_to(WORKSPACE)}")

    rows, ok_all = [], True
    print(f"\n{'shape':16s} {'candidate':24s} {'bug':10s} {'VOLTA':13s} {'expected':22s} {'ok':3s} {'wall s':>7}  detail")
    print("-" * 132)
    for shape in shapes:
        for c in cands:
            r = run_compare(make_case(args.op, builds[ref.name].ptx, ref, builds[c.name].ptx, c, shape, args.timeout))
            exp = expected_statuses(c.expect_volta, shape)
            ok = exp is None or r.status in exp
            ok_all &= ok
            cat = (c.bug or {}).get("category", "-")
            exp_s = "|".join(exp) if exp else "-"
            print(f"{shape_label(shape):16s} {c.name:24s} {cat:10s} {r.status:13s} {exp_s[:22]:22s} "
                  f"{'yes' if ok else 'NO':3s} {r.wall_s:7.2f}  {describe(r)[:60]}")
            rows.append({"shape": shape, "candidate": c.name, "bug": c.bug, "expected": exp,
                         "matches_expectation": ok, **r.to_dict()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"{args.op}_{stamp}.json"
    out.write_text(json.dumps({"created": stamp, "op": args.op, "shapes": shapes, "reference": ref.name,
                               "tools": tool_versions(), "compile": compile_log, "results": rows}, indent=2))
    print(f"\nsaved {out.relative_to(WORKSPACE)}   all as expected: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
