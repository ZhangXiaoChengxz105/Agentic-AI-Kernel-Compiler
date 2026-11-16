"""VOLTA on every CUDA variant of a workload x shapes, checked against the manifest's expect_volta.

    akc volta --workload gemm [--shapes "M=32,N=32,K=40;M=12,N=12,K=40"] [--only a,b] [--timeout 600]
Output: table + results/volta/<op>_<ts>.json; exit 0 iff every verdict matched its expectation. CPU only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

from ..backend.ptx import compile_to_ptx
from ..paths import BUILD, RESULTS, THIRD_PARTY
from ..task.workload import list_workloads, load_workload
from ..validation.volta.layouts import expected_statuses, make_case, shape_label
from ..validation.volta.runner import run_compare


def parse_shapes(text: str) -> list[dict]:
    return [{k.strip(): int(v) for k, v in (kv.split("=") for kv in chunk.split(","))}
            for chunk in text.split(";") if chunk.strip()]


def _describe(r) -> str:
    if r.status == "EQUIV":
        return f"{r.elements_checked} elements proved equal"
    if r.status == "NOT_EQUIV":
        return f"{r.mismatches} elements differ, e.g. {', '.join(r.mismatch_examples[:3])}"
    return (f"[{r.which}] " if r.which else "") + r.detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="akc volta", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=list_workloads())
    ap.add_argument("--shapes", help='e.g. "N=256;N=200" (default: manifest volta default + deep + boundary shapes)')
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)

    wl = load_workload(a.workload)
    ref = wl.reference("cuda")
    cands = [v for v in wl.candidates("cuda") if not a.only or v.name in a.only.split(",")]
    if a.shapes:
        shapes = parse_shapes(a.shapes)
    else:
        from ..validation.changeaware.generate import volta_shapes
        shapes = volta_shapes(a.workload, ["tiling"])
    out_dir = BUILD / "ptx" / a.workload
    ptx = {v.name: compile_to_ptx(v.file, out_dir / v.name, mode="volta").ptx for v in [ref, *cands]}
    rows, ok_all = [], True
    print(f"{'shape':16s} {'candidate':24s} {'bug':10s} {'VOLTA':13s} {'expected':22s} {'ok':3s} {'wall s':>7}  detail")
    for s in shapes:
        for c in cands:
            r = run_compare(make_case(a.workload, ptx[ref.name], ref, ptx[c.name], c, s, a.timeout))
            exp = expected_statuses(c.expect_volta, s)
            ok = exp is None or r.status in exp
            ok_all &= ok
            print(f"{shape_label(s):16s} {c.name:24s} {(c.bug or {}).get('category', '-'):10s} {r.status:13s} "
                  f"{('|'.join(exp) if exp else '-')[:22]:22s} {'yes' if ok else 'NO':3s} {r.wall_s:7.2f}  {_describe(r)[:60]}")
            rows.append({"shape": s, "candidate": c.name, "bug": c.bug, "expected": exp, "matches_expectation": ok, **r.to_dict()})
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=THIRD_PARTY / "volta").stdout.strip()
    d = RESULTS / "volta"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{a.workload}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"workload": a.workload, "shapes": shapes, "volta_commit": commit, "results": rows}, indent=2))
    print(f"\nsaved {f}   all as expected: {ok_all}")
    return 0 if ok_all else 1
