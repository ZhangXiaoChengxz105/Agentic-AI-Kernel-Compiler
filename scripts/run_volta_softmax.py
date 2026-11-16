#!/usr/bin/env python3
"""Building block 4: end-to-end VOLTA loop for the softmax kernels.

  for every variant in kernels/softmax/manifest.json:  .cu --(verify.ptx, mode=volta)--> .ptx
  for every N in --sizes, for every candidate:          volta compare(ref, candidate) --> VoltaResult
  print a table, compare against the manifest's expect_volta, save everything as JSON.

Usage:
  source env.sh
  envs/tirx/bin/python scripts/run_volta_softmax.py [--sizes 256,200] [--timeout 600]
Output:
  stdout table; results/volta/softmax_<timestamp>.json; PTX files under build/ptx/softmax/.
Exit code: 0 if every result matched expectations, 1 otherwise.
CPU only (no GPU, no Lumen). Small N runs in well under a second per pair.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import WORKSPACE  # noqa: E402
from verify.cases import SOFTMAX_DIR, load_manifest, softmax_case  # noqa: E402
from verify.ptx import compile_to_ptx  # noqa: E402
from verify.volta import run_compare  # noqa: E402


def expected_status(expect, n: int, block: int) -> str | None:
    if expect is None or isinstance(expect, str):
        return expect
    return expect["divisible"] if n % block == 0 else expect["not_divisible"]


def tool_versions() -> dict:
    def run(cmd, cwd=None):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd).stdout.strip()
        except OSError as e:
            return f"unavailable: {e}"
    nvcc = run(["nvcc", "--version"]).splitlines()
    return {
        "nvcc": nvcc[-1] if nvcc else "?",
        "volta_commit": run(["git", "rev-parse", "--short", "HEAD"], cwd=WORKSPACE / "third_party" / "volta"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="256,200", help="comma-separated row lengths N")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per VOLTA run")
    ap.add_argument("--build-dir", default=str(WORKSPACE / "build" / "ptx" / "softmax"))
    ap.add_argument("--out-dir", default=str(WORKSPACE / "results" / "volta"))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    meta, variants = load_manifest(SOFTMAX_DIR)
    ref = next(v for v in variants if v.role == "reference")
    cands = [v for v in variants if v.role == "candidate"]

    # 1) compile every variant for VOLTA
    builds, compile_log = {}, []
    for v in variants:
        b = compile_to_ptx(v.file, args.build_dir, mode="volta")
        if v.name not in b.kernels:
            raise SystemExit(f"{v.file}: kernel {v.name} not found in PTX entries {b.kernels}")
        builds[v.name] = b
        compile_log.append({"variant": v.name, "ptx": str(b.ptx), "seconds": round(b.seconds, 3),
                            "cmd": list(b.cmd), "warnings": b.warnings})
        print(f"[compile] {v.name:24s} -> {b.ptx.relative_to(WORKSPACE)}  ({b.seconds:.2f}s)")

    # 2) run VOLTA on every (N, candidate)
    rows, ok_all = [], True
    print()
    print(f"{'N':>5}  {'candidate':24s} {'bug category':12s} {'VOLTA status':13s} {'expected':10s} {'ok':3s} "
          f"{'wall s':>7}  detail")
    print("-" * 120)
    for n in sizes:
        for c in cands:
            case = softmax_case(builds[ref.name].ptx, ref, builds[c.name].ptx, c, n, timeout_s=args.timeout)
            r = run_compare(case)
            exp = expected_status(c.expect_volta, n, c.block)
            ok = (exp is None) or (r.status == exp)
            ok_all &= ok
            cat = (c.bug or {}).get("category", "-")
            detail = r.detail if r.status != "NOT_EQUIV" else f"{r.mismatches} elements differ, e.g. {', '.join(r.mismatch_examples[:3])}"
            if r.status == "EQUIV":
                detail = f"{r.elements_checked} elements proved equal"
            print(f"{n:>5}  {c.name:24s} {cat:12s} {r.status:13s} {str(exp):10s} {'yes' if ok else 'NO':3s} "
                  f"{r.wall_s:7.2f}  {detail[:70]}")
            rows.append({"N": n, "candidate": c.name, "bug": c.bug, "expected": exp, "matches_expectation": ok,
                         **r.to_dict()})

    # 3) save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"softmax_{stamp}.json"
    out.write_text(json.dumps({
        "created": stamp, "op": meta["op"], "sizes": sizes, "reference": ref.name,
        "tools": tool_versions(), "compile": compile_log, "results": rows,
    }, indent=2))
    print(f"\nsaved {out.relative_to(WORKSPACE)}   all as expected: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
