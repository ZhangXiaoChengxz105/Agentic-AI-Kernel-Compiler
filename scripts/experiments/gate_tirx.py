"""D2 validation: the compiler-checks gate (akc/validation/compiler_checks/) on tirx-lite kernels. This is the first
time compute-sanitizer runs on TIRx-generated code.

Kernels (all tirx-lite):
  ref_opt   the correct references and optimized kernels: softmax_txl_ref, layernorm_txl_{ref,opt},
            gemm_txl_{ref,opt}                                                     any report = a false alarm
  glm       the 4 GLM-written tirx softmax kernels (glm_txl_C_r2, D1_r2, D2_r1, D3_r1), all correct
                                                                                   any report = a false alarm
  injected  the 11 injected tirx bugs (softmax 4, layernorm 4, gemm 3)
  mutant    the 14 tirx mutants the tests missed, from results/eval/volta_tirx.json (group "mutant"), regenerated with
            akc.evaluation.mutants.mutants(src, "tirx", 2); 8 are races / out-of-bounds by VOLTA (the targets), 6 are
            EQUIV by VOLTA (reported separately: VOLTA checks CTA 0 at 3 sizes only, so a report there is not
            automatically a false alarm)
Each kernel runs through CompilerChecksValidator().validate(cand, ValidationContext(device=...)), i.e. exactly the
gate of configurations A / B / C: static checks, then the 4 sanitizer tools in parallel (GPU) or NumSim (CPU).

Usage:
  python scripts/experiments/gate_tirx.py [--device gpu|cpu] [--groups ref_opt,glm,injected,mutant] [--labels a,b]
         [--sequential softmax_txl_ref,gemm_txl_opt] [--out results/eval/gate_tirx.json]
  --sequential L: also run these kernels with the 4 tools one after another (cost comparison).
Output: results/eval/gate_tirx.json: meta, rows (per kernel: group, expectation, caught, per-tool caught / kinds /
        reports / seconds, stage seconds, value check, findings), summary (per group and tool: caught / total,
        false alarms, seconds).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import BASES, HELDOUT, mutants  # noqa: E402
from akc.paths import RESULTS, ROOT  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation.compiler_checks import CompilerChecksValidator  # noqa: E402
from akc.validation.interface import Candidate, ValidationContext  # noqa: E402

GROUPS = ("ref_opt", "glm", "injected", "mutant")
CORRECT_GROUPS = ("ref_opt", "glm")
GLM = ("glm_txl_C_r2", "glm_txl_D1_r2", "glm_txl_D2_r1", "glm_txl_D3_r1")
FAIL = ("NOT_EQUIV", "RACE", "DEADLOCK", "OUT_OF_BOUNDS", "UNINIT", "FOOTPRINT", "DIV_BY_ZERO")


def kernels(groups) -> list[dict]:
    out = []
    if "ref_opt" in groups:
        for wl_name in ("softmax", "layernorm", "gemm"):
            wl = load_workload(wl_name)
            for v in wl.variants:
                if v.lang == "tirx" and not v.bug:
                    out.append({"label": v.name, "group": "ref_opt", "workload": wl_name, "src": v.file.read_text(),
                                "path": str(v.file), "expect": "clean"})
    where = {lab: (wl, ROOT / rel) for lab, wl, rel in BASES + HELDOUT}
    if "glm" in groups:
        for lab in GLM:
            wl, p = where[lab]
            out.append({"label": lab, "group": "glm", "workload": wl, "src": p.read_text(), "path": str(p),
                        "expect": "clean"})
    if "injected" in groups:
        for wl_name in ("softmax", "layernorm", "gemm"):
            wl = load_workload(wl_name)
            for v in wl.bugs("tirx"):
                out.append({"label": v.name, "group": "injected", "workload": wl_name, "src": v.file.read_text(),
                            "path": str(v.file), "bug": v.bug, "expect_volta": v.expect_volta, "expect": "bug"})
    if "mutant" in groups:
        rows = json.loads((RESULTS / "eval" / "volta_tirx.json").read_text())["rows"]
        for r in rows:
            if r["group"] != "mutant":
                continue
            wl, p = where[r["base"]]
            ms = [m for m in mutants(p.read_text(), "tirx", 2) if m.op == r["op"] and m.line == r["line"]
                  and m.desc == r["desc"]]
            if len(ms) != 1:
                raise RuntimeError(f"cannot regenerate {r['kernel']} ({len(ms)} matches)")
            statuses = sorted({x["status"] for x in r["runs"]})
            harmful = bool(set(statuses) & set(FAIL))
            out.append({"label": r["kernel"], "group": "mutant", "workload": wl, "src": ms[0].src, "path": None,
                        "base": r["base"], "op": r["op"], "line": r["line"], "desc": r["desc"], "volta": statuses,
                        "expect": "bug" if harmful else "volta_equiv"})
    return out


def run_one(k: dict, device: str, parallel: bool = True) -> dict:
    cand = Candidate(k["workload"], "tirx", k["src"], label=k["label"],
                     path=Path(k["path"]) if k.get("path") else None)
    t0 = time.perf_counter()
    rep = CompilerChecksValidator().validate(cand, ValidationContext(device=device, extra={"gate_parallel": parallel}))
    wall = time.perf_counter() - t0
    art = rep.artifacts
    tools = art.get("sanitizer") or {}
    stage_s = {s["stage"]: s["seconds"] for s in art.get("stages", [])}
    return {**{x: k[x] for x in k if x not in ("src",)}, "device": device, "parallel": parallel,
            "caught": not rep.passed, "decided": rep.decided, "seconds": round(wall, 2),
            "kinds": sorted({f.kind for f in rep.findings if f.severity == "error"}),
            "static_error": any(f.kind == "compile_error" for f in rep.findings),
            "tools": {t: {kk: v.get(kk) for kk in ("caught", "kinds", "n_reports", "by_severity", "seconds", "exit",
                                                   "log", "completed", "verdict", "not_ours", "api_errors",
                                                   "kernel_fault")}
                      for t, v in tools.items()},
            "stages": stage_s, "sanitizer_wall_s": art.get("sanitizer_wall_s"), "shapes": art.get("shapes"),
            "value_check": art.get("value_check"),
            "findings": [{"kind": f.kind, "severity": f.severity, "message": f.message[:300], "location": f.location,
                          "cuda": (f.evidence.get("cuda") or {}).get("code"), "tool": f.evidence.get("tool")}
                         for f in rep.findings][:12]}


def summarize(rows: list[dict]) -> dict:
    s: dict = {}
    for r in rows:
        if r.get("sequential_rerun"):
            continue
        g = s.setdefault(r["group"] + ("" if r["group"] != "mutant" else f"/{r['expect']}"), {"n": 0, "any": 0, "ids": [],
                                                                                              "tools": {}})
        g["n"] += 1
        if r["caught"]:
            g["any"] += 1
            g["ids"].append(r["label"])
        for t, v in r["tools"].items():
            d = g["tools"].setdefault(t, {"caught": 0, "seconds": []})
            d["caught"] += bool(v.get("caught"))
            if v.get("seconds") is not None:
                d["seconds"].append(v["seconds"])
        if r["static_error"]:
            g.setdefault("static_errors", []).append(r["label"])
    for g in s.values():
        for d in g["tools"].values():
            sec = d.pop("seconds")
            d["median_s"] = round(statistics.median(sec), 2) if sec else None
    main = [r for r in rows if not r.get("sequential_rerun")]
    tot = [r["seconds"] for r in main]
    st = {}
    for r in main:
        for k, v in r["stages"].items():
            st.setdefault(k, []).append(v)
    s["cost"] = {"gate_seconds_median": round(statistics.median(tot), 2) if tot else None,
                 "gate_seconds_max": round(max(tot), 2) if tot else None,
                 "stage_seconds_median": {k: round(statistics.median(v), 3) for k, v in st.items()},
                 "sanitizer_wall_median": round(statistics.median([r["sanitizer_wall_s"] for r in main
                                                                   if r.get("sanitizer_wall_s")]), 2)
                 if any(r.get("sanitizer_wall_s") for r in main) else None}
    seq = [r for r in rows if r.get("sequential_rerun")]
    if seq:
        s["cost"]["sequential_vs_parallel"] = [
            {"label": r["label"], "sequential_sanitizer_s": r["sanitizer_wall_s"],
             "parallel_sanitizer_s": next((p["sanitizer_wall_s"] for p in main if p["label"] == r["label"]), None),
             "same_verdict": r["caught"] == next((p["caught"] for p in main if p["label"] == r["label"]), None)}
            for r in seq]
    s["false_alarms"] = sum(s.get(g, {}).get("any", 0) for g in CORRECT_GROUPS)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--labels", default="")
    ap.add_argument("--sequential", default="", help="labels to re-run with the 4 tools one after another")
    ap.add_argument("--out", default=str(RESULTS / "eval" / "gate_tirx.json"))
    a = ap.parse_args(argv)
    ks = kernels([g for g in a.groups.split(",") if g])
    if a.labels:
        keep = set(a.labels.split(","))
        ks = [k for k in ks if k["label"] in keep]
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename, "device": a.device, "groups": a.groups, "n_kernels": len(ks)}
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
        from akc.validation.compiler_checks.sanitizer import SANITIZER_FLAGS, find_compute_sanitizer
        cs = find_compute_sanitizer()
        meta["compute_sanitizer"] = cs
        meta["sanitizer_version"] = subprocess.run([cs, "--version"], capture_output=True, text=True).stdout.strip(
        ).splitlines()[-1] if cs else None
        meta["sanitizer_flags"] = SANITIZER_FLAGS
    except Exception as e:  # noqa: BLE001
        meta["gpu"] = f"unknown ({e})"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.perf_counter()

    def save(done=False):
        out.write_text(json.dumps({"meta": {**meta, "wall_s": round(time.perf_counter() - t_start, 1), "complete": done},
                                   "rows": rows, "summary": summarize(rows)}, indent=1, default=str))
    print(f"{len(ks)} kernels on {a.device}", flush=True)
    for k in ks:
        r = run_one(k, a.device)
        rows.append(r)
        tools = " ".join(f"{t.split(':')[-1]}={'X' if v.get('caught') else '.'}{v.get('seconds') or 0:.0f}s"
                         for t, v in r["tools"].items())
        print(f"{r['group']:9s} {r['label']:34s} expect={r['expect']:11s} {'CAUGHT' if r['caught'] else 'clean ':6s} "
              f"{','.join(r['kinds']):28s} {r['seconds']:6.1f}s  {tools}  "
              f"values={[v.get('reason') for v in r.get('value_check') or []]}", flush=True)
        for f in r["findings"][:3]:
            print(f"      {f['severity']:7s} {f['kind']:14s} {str(f['location'])[:60]:60s} {f['message'][:120]}", flush=True)
        save()
    for lab in [x for x in a.sequential.split(",") if x]:
        k = next((k for k in ks if k["label"] == lab), None)
        if k is None:
            continue
        r = run_one(k, a.device, parallel=False)
        r["sequential_rerun"] = True
        rows.append(r)
        print(f"sequential {lab}: sanitizer {r['sanitizer_wall_s']}s (tools {[(t, v.get('seconds')) for t, v in r['tools'].items()]})",
              flush=True)
        save()
    save(True)
    print(json.dumps(summarize(rows), indent=1, default=str))
    print(f"saved {out} ({time.perf_counter() - t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
