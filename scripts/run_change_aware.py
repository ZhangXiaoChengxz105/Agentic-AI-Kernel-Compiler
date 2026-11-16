#!/usr/bin/env python3
"""Task 2 end-to-end on the injected bugs (CPU only: diff -> classify -> generate -> VOLTA).

For every (opt -> bug) pair in kernels/*/manifest.json:
  1. changeaware.plan(): hunks, labels (with evidence), categories, targeted GPU tests, VOLTA shapes
  2. VOLTA as configuration B would run it: one fixed default shape
  3. VOLTA as configuration C would run it: the change-aware shapes
  4. save the generated GPU tests to results/change_aware/tests/<bug>.json
     (run later on a GPU: python -m harness.cli check --op <op> --variants <bug> --tests <file>)
Prints a summary table and writes results/change_aware/summary_<ts>.json.

Usage: source env.sh && envs/tirx/bin/python scripts/run_change_aware.py [--ops softmax,gemm,layernorm] [-v]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from changeaware import plan  # noqa: E402
from changeaware.generate import DEFAULT_VOLTA  # noqa: E402
from verify import WORKSPACE  # noqa: E402
from verify.cases import FAIL_STATUSES, load_manifest, make_case, shape_label  # noqa: E402
from verify.ptx import compile_to_ptx  # noqa: E402
from verify.volta import run_compare  # noqa: E402


def volta_verdict(op, ref, ref_ptx, cand, cand_ptx, shapes, timeout):
    """Run VOLTA on each shape; return (detected?, first failing shape/status, per-shape list)."""
    runs = []
    for s in shapes:
        r = run_compare(make_case(op, ref_ptx, ref, cand_ptx, cand, s, timeout))
        runs.append({"shape": s, "status": r.status, "wall_s": round(r.wall_s or 0, 3), "detail": r.detail[:160]})
    hit = next((x for x in runs if x["status"] in FAIL_STATUSES), None)
    return hit is not None, hit, runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", default="softmax,gemm,layernorm")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("-v", "--verbose", action="store_true", help="print hunks/labels/tests per pair")
    args = ap.parse_args()
    out_dir = WORKSPACE / "results" / "change_aware"
    (out_dir / "tests").mkdir(parents=True, exist_ok=True)

    rows = []
    hdr = f"{'bug':24s} {'truth':9s} {'predicted':20s} {'#tests':>6s}  {'VOLTA (B: fixed shape)':26s} {'VOLTA (C: change-aware)':34s}"
    print(hdr + "\n" + "-" * len(hdr))
    for op in args.ops.split(","):
        _, variants = load_manifest(op)
        ref = next(v for v in variants if v.role == "reference")
        opt = next(v for v in variants if v.name == f"{op}_opt")
        bdir = WORKSPACE / "build" / "ptx" / op
        ptx = {v.name: compile_to_ptx(v.file, bdir, mode="volta").ptx for v in variants}
        for bug in [v for v in variants if v.bug]:
            p = plan(op, opt.file.read_text(), bug.file.read_text(), bug.block)
            b_det, b_hit, b_runs = volta_verdict(op, ref, ptx[ref.name], bug, ptx[bug.name], [DEFAULT_VOLTA[op]], args.timeout)
            c_det, c_hit, c_runs = volta_verdict(op, ref, ptx[ref.name], bug, ptx[bug.name], p.volta, args.timeout)
            tests_file = out_dir / "tests" / f"{bug.name}.json"
            tests_file.write_text(json.dumps([t.to_dict() for t in p.tests], indent=1))
            b_s = f"{b_hit['status']} @{shape_label(b_hit['shape'])}" if b_det else f"missed ({b_runs[0]['status']})"
            c_s = f"{c_hit['status']} @{shape_label(c_hit['shape'])}" if c_det else f"missed ({', '.join(sorted({r['status'] for r in c_runs}))})"
            print(f"{bug.name:24s} {bug.bug['category']:9s} {','.join(p.cats):20s} {len(p.tests):6d}  {b_s:26s} {c_s:34s}")
            if args.verbose:
                for lab in p.labels:
                    print(f"      label {lab.category:9s} {lab.rule:15s} L{lab.line}: {lab.evidence[:80]}")
                for t in p.tests:
                    print(f"      test  {t.key():58s} {t.tag}")
                print(f"      volta shapes: {[shape_label(s) for s in p.volta]}")
            rows.append({"op": op, "bug": bug.name, "truth": bug.bug["category"], "predicted": p.cats,
                         "labels": [vars(lab) for lab in p.labels], "tests_file": str(tests_file.relative_to(WORKSPACE)),
                         "n_tests": len(p.tests), "volta_B": {"detected": b_det, "runs": b_runs},
                         "volta_C": {"detected": c_det, "runs": c_runs}})
    n = len(rows)
    print(f"\ncategory detected: {sum(r['truth'] in r['predicted'] for r in rows)}/{n}   "
          f"VOLTA detection  B (fixed shape): {sum(r['volta_B']['detected'] for r in rows)}/{n}   "
          f"C (change-aware shapes): {sum(r['volta_C']['detected'] for r in rows)}/{n}")
    p = out_dir / f"summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(rows, indent=2))
    print(f"saved {p.relative_to(WORKSPACE)}  (GPU test lists in results/change_aware/tests/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
