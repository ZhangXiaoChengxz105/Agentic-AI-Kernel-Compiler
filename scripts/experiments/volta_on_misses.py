"""Run VOLTA on the CUDA mutants that configuration C's tests missed (from `akc mutants --detect`).

Why: the misses were sorted by reading the code into "cannot change outputs here", "only the max shift changes"
and "race / out-of-bounds shared access (tests cannot see it)". VOLTA checks both claims: EQUIV over the reals
supports "equivalent", RACE / OUT_OF_BOUNDS shows that full configuration C (tests + VOLTA) catches the edit.

Usage (CPU only, a few minutes; run inside a Slurm job, not on the login node):
  python scripts/experiments/volta_on_misses.py results/eval/mutants_detect_bases.json results/eval/mutants_detect_heldout.json
Output: results/eval/mutants_volta_misses.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import BASES, HELDOUT, mutants  # noqa: E402
from akc.paths import RESULTS, ROOT  # noqa: E402
from akc.validation.changeaware import plan  # noqa: E402
from akc.validation.interface import Candidate, ValidationContext  # noqa: E402
from akc.validation.volta.validator import VoltaValidator  # noqa: E402


def main(paths: list[str]) -> int:
    where = {label: (wl, rel) for label, wl, rel in BASES + HELDOUT}
    rows, t0 = [], time.perf_counter()
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            det = r.get("detect")
            if r["lang"] != "cuda" or not det or det["C"]["caught"] or not r["accept"] or not r.get("bug", True):
                continue
            wl, rel = where[r["base"]]
            src = (ROOT / rel).read_text()
            m = next(m for m in mutants(src, "cuda", 2) if m.op == r["op"] and m.line == r["line"])
            shapes = plan(wl, src, m.src, "cuda").volta
            cand = Candidate(wl, "cuda", m.src, label=f"{r['base']}_{m.op}_{m.line}")
            rep = VoltaValidator(timeout_s=120).validate(cand, ValidationContext(volta_shapes=shapes))
            statuses = [run["status"] for run in rep.artifacts.get("runs", [])] or [f.kind for f in rep.findings]
            row = {"base": r["base"], "op": m.op, "line": m.line, "desc": m.desc, "shapes": shapes,
                   "statuses": statuses, "caught": not rep.passed, "seconds": round(rep.seconds, 1)}
            rows.append(row)
            print(f"{r['base']:14s} {m.op:8s} L{m.line:<4d} {m.desc[:40]:40s} {','.join(sorted(set(statuses)))}", flush=True)
    out = RESULTS / "eval" / "mutants_volta_misses.json"
    out.write_text(json.dumps({"rows": rows, "seconds": time.perf_counter() - t0}, indent=1))
    print(f"VOLTA caught {sum(r['caught'] for r in rows)}/{len(rows)} of the CUDA mutants C's tests missed; saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
