"""Re-run A0/A/B/C on the tirx-lite mutants whose change-aware tests included misaligned pointers.

Why: a function compiled by TIRx checks the data alignment of every tensor argument and raises an error for a
misaligned one, so the change-aware "pointers misaligned by 1 element" test failed for every tirx kernel (correct or
not) and was counted as a catch. The generator no longer emits that test for tirx-lite (generate.py, memory). Only
mutants labelled "memory" got it, so only those rows are re-run; every other row is unchanged.

    python scripts/experiments/rerun_mutants.py --confirm               # show the error on a correct kernel (GPU)
    python scripts/experiments/rerun_mutants.py --set heldout --volta-workers 6

The set's file results/eval/mutants_detect_tirx_<set>.json is first copied to ..._before_misalign_fix.json (once),
then the re-run rows replace the old ones in place, each with "rerun": {job, reason, old: {config: caught}}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import FRESH_LABELS, PER_OP, SETS, _cands, _detect, mutants  # noqa: E402
from akc.paths import RESULTS, ROOT  # noqa: E402

REASON = "change-aware tests had a misaligned-pointer test, which a TIRx-compiled function always refuses"


def confirm() -> None:
    from akc.task.testcase import TestCase
    from akc.task.workload import load_workload
    from akc.validation.runners.gpu import run_tests_isolated
    wl = load_workload("softmax")
    ref = wl.reference("tirx")
    tests = [TestCase("softmax", {"rows": 3, "N": 1024}, "normal", 1, 1, "aligned"),
             TestCase("softmax", {"rows": 3, "N": 1024}, "normal", 1, 1, "misaligned by 1 element", 1)]
    for r in run_tests_isolated("softmax", "tirx", ref.file, None, ref.block, tests):
        print(f"{ref.name} {r.get('tag')}: passed={r.get('passed')} reason={r.get('reason')} "
              f"message={str(r.get('message'))[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--set", choices=["bases", "heldout", "fresh"])
    ap.add_argument("--volta-workers", type=int, default=6)
    a = ap.parse_args()
    if a.confirm:
        confirm()
        return 0
    f = RESULTS / "eval" / f"mutants_detect_tirx_{a.set}.json"
    backup = f.with_name(f.stem + "_before_misalign_fix.json")
    if not backup.exists():
        shutil.copy(f, backup)
    d = json.loads(f.read_text())
    todo = [i for i, r in enumerate(d["rows"]) if "detect" in r and "memory" in (r.get("categories") or [])]
    print(f"{a.set}: re-running {len(todo)} of {len(d['rows'])} rows", flush=True)
    from akc.evaluation.volta_pool import VoltaPool
    pool = VoltaPool(a.volta_workers)
    srcs = {label: ((ROOT / rel).read_text(), workload) for label, workload, rel in SETS[a.set]}
    muts = {}
    for i in todo:
        r = d["rows"][i]
        if r["base"] not in muts:
            src = srcs[r["base"]][0]
            muts[r["base"]] = {(m.op, m.line, m.desc): m for m in
                               mutants(src, "tirx", PER_OP, extended=r["base"] in FRESH_LABELS)}
        m = muts[r["base"]][(r["op"], r["line"], r["desc"])]
        pool.submit(_cands(r["workload"], r["base"], srcs[r["base"]][0], "tirx", m))
    t0 = time.perf_counter()
    for i in todo:
        r = d["rows"][i]
        src, workload = srcs[r["base"]]
        m = muts[r["base"]][(r["op"], r["line"], r["desc"])]
        old = {c: r["detect"][c]["caught"] for c in ("A0", "A", "B", "C") if c in r["detect"]}
        r["detect"] = _detect(workload, r["base"], src, "tirx", m, "auto", True, None, "auto", pool)
        r["rerun"] = {"job": os.environ.get("SLURM_JOB_ID"), "reason": REASON, "old": old}
        new = {c: r["detect"][c]["caught"] for c in old}
        print(f"  {r['base']} {r['op']} L{r['line']} {r['desc'][:40]}: old {old} new {new} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        f.write_text(json.dumps(d, indent=1))
    pool.shutdown()
    d.setdefault("reruns", []).append({"job": os.environ.get("SLURM_JOB_ID"), "rows": len(todo), "reason": REASON})
    f.write_text(json.dumps(d, indent=1))
    print(f"saved {f} (before: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
