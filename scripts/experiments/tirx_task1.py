"""Task 1 baseline for the tirx-lite LayerNorm and GEMM kernels: GPU correctness, NumSim, VOLTA, speed.

Kernels: every tirx variant of layernorm and gemm in the manifests (reference, the optimized candidate, the injected
bugs ported from the CUDA ones). For each kernel:
  gpu     the fixed suite (3 repeats, N(0,1)) plus change-aware-style tests inside the declared shape domain:
          non-divisible sizes (N % 128 != 0; M, N, K % 16 != 0), size 1, the domain edge, and the op's numerical
          distributions (TESTS below). Runner: runners.gpu.run_tests_isolated (lang "tirx"; sentinel / canary /
          repeat checks, one child process per kernel).                                                [GPU]
  numsim  the same kernel on the CPU with TIRx NumSim + Racecheck at one small non-divisible shape, two value
          distributions (NUMSIM_TESTS; the first test also runs Racecheck).                            [CPU]
  volta   scripts/experiments/volta_tirx.py --groups reference,opt,bug at the manifest VOLTA default / deep shapes
          plus the change-aware tiling / sync VOLTA shapes; softmax reference + tirx bugs too, to check their
          new manifest expect_volta. Writes results/eval/volta_tirx_{layernorm,gemm,softmax_task1}.json.  [CPU]
  perf    perf.measure (correctness at the benchmark shape, then CUDA-event timing, L2 flushed, median of 50)
          of the tirx reference and optimized kernel vs PyTorch at manifest perf.shape.                  [GPU]
"caught" for a bug = at least one GPU test fails, or NumSim fails, or VOLTA reports an error verdict at a shape.

Usage (in a Slurm GPU job; numsim and volta run on the CPU in background threads while gpu / perf run):
  python scripts/experiments/tirx_task1.py [--parts gpu,numsim,volta,perf] [--workloads layernorm,gemm]
Output: results/eval/tirx_task1.json (merged per kernel / workload: what was not re-run keeps its previous content),
        results/perf/tirx_task1_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from akc.paths import RESULTS  # noqa: E402
from akc.task.testcase import TestCase  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402

OUT = RESULTS / "eval" / "tirx_task1.json"


def _t(op, shapes_dists, tag, repeat=1, seed0=100):
    return [TestCase(op, s, d, seed=seed0 + i, repeat=repeat, tag=tag) for i, (s, d) in enumerate(shapes_dists)]


def gpu_tests(op: str) -> list[TestCase]:
    from akc.validation.fixed_tests.suite import fixed_suite
    fixed = fixed_suite(op)
    if op == "layernorm":
        L = lambda r, n: {"rows": r, "N": n}  # noqa: E731
        extra = _t(op, [(L(8, 1), "normal"), (L(8, 127), "normal"), (L(8, 129), "normal"), (L(8, 200), "normal"),
                        (L(4, 4097), "normal"), (L(2, 16384), "normal")], "tiling: N % 128 != 0 / edge")
        extra += _t(op, [(L(16, 1000), "normal")], "sync: repeated", repeat=5, seed0=120)
        extra += _t(op, [(L(64, 1000), "offset"), (L(16, 1000), "large"), (L(16, 1000), "tiny"),
                         (L(16, 1000), "constant"), (L(256, 1000), "outlier@128")], "numerical", seed0=130)
    elif op == "gemm":
        G = lambda m, n, k: {"M": m, "N": n, "K": k}  # noqa: E731
        extra = _t(op, [(G(1, 1, 1), "normal"), (G(12, 12, 40), "normal"), (G(15, 17, 13), "normal"),
                        (G(17, 15, 40), "normal"), (G(33, 33, 33), "normal"), (G(128, 128, 40), "normal"),
                        (G(257, 129, 1000), "normal"), (G(4096, 17, 33), "normal")], "tiling: % 16 != 0 / edge")
        extra += _t(op, [(G(100, 100, 100), "normal"), (G(16, 16, 64), "normal")], "sync: repeated", repeat=5,
                    seed0=120)
        extra += _t(op, [(G(64, 64, 64), "large"), (G(64, 64, 64), "mixed"), (G(64, 64, 64), "tiny"),
                         (G(64, 64, 64), "outlier@16")], "numerical", seed0=130)
    else:
        raise ValueError(op)
    return fixed + extra


NUMSIM_TESTS = {
    "layernorm": [({"rows": 2, "N": 200}, "normal"), ({"rows": 2, "N": 200}, "offset")],
    "gemm": [({"M": 12, "N": 12, "K": 40}, "normal"), ({"M": 12, "N": 12, "K": 40}, "large")],
}


def kernels(workloads: list[str]):
    out = []
    for op in workloads:
        wl = load_workload(op)
        for v in wl.variants:
            if v.lang == "tirx":
                out.append((op, v))
    return out


def _brief(outcomes: list[dict]) -> dict:
    fails = [o for o in outcomes if not o["passed"] and o["reason"] != "skipped"]
    return {"passed": sum(o["passed"] for o in outcomes), "failed": len(fails),
            "skipped": sum(o["reason"] == "skipped" for o in outcomes), "total": len(outcomes),
            "reasons": sorted({o["reason"] for o in fails}),
            "failing_tests": [{"shape": o["shape"], "dist": o["dist"], "repeat": o["repeat"], "tag": o["tag"],
                               "reason": o["reason"], "message": (o.get("message") or "")[:200]} for o in fails]}


def run_gpu(ks, workers: int) -> dict:
    from akc.validation.runners.gpu import run_tests_isolated

    def one(item):
        op, v = item
        t0 = time.perf_counter()
        outs = run_tests_isolated(op, "tirx", str(v.file), None, v.block, gpu_tests(op), timeout_s=1200)
        b = _brief(outs)
        b["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"[gpu] {v.name:28s} {b['passed']}/{b['total']} passed  {','.join(b['reasons'])}  ({b['seconds']}s)",
              flush=True)
        return v.name, {"summary": b, "outcomes": outs}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, ks))


def run_numsim(ks, workers: int) -> dict:
    from akc.validation.runners.numsim import run_tests_numsim_isolated

    def one(item):
        op, v = item
        tests = [TestCase(op, s, d, seed=200 + i, tag="sync: numsim" if i == 0 else "numsim")
                 for i, (s, d) in enumerate(NUMSIM_TESTS[op])]
        t0 = time.perf_counter()
        outs = run_tests_numsim_isolated(op, v.file, tests, timeout_s=1500)
        b = _brief(outs)
        b["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"[numsim] {v.name:28s} {b['passed']}/{b['total']} passed  {','.join(b['reasons'])}  ({b['seconds']}s)",
              flush=True)
        return v.name, {"summary": b, "outcomes": [{k: o.get(k) for k in ("shape", "dist", "passed", "reason",
                                                                              "message", "sim_seconds")} for o in outs]}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, ks))


def run_volta(workloads: list[str]) -> dict:
    out = {}
    jobs = [(w, "reference,opt,bug", RESULTS / "eval" / f"volta_tirx_{w}.json") for w in workloads]
    jobs.append(("softmax", "reference,bug", RESULTS / "eval" / "volta_tirx_softmax_task1.json"))
    for w, groups, path in jobs:
        cmd = [sys.executable, str(ROOT / "scripts" / "experiments" / "volta_tirx.py"), "--workload", w,
               "--groups", groups, "--out", str(path), "--timeout", "600"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print("".join(f"[volta] {ln}\n" for ln in p.stdout.splitlines() if not ln.startswith((" ", "{", "}"))),
              end="", flush=True)
        if p.returncode != 0 or not path.exists():
            out[w] = {"error": (p.stderr or p.stdout)[-2000:]}
            continue
        d = json.loads(path.read_text())
        out[w] = {"file": str(path.relative_to(ROOT)), "summary": d["summary"],
                  "kernels": {r["kernel"]: {"decided": r["decided"], "caught": r["caught"],
                                            "runs": [{"shape": x["shape"], "status": x["status"],
                                                      "expect": x["expect"], "as_expected": x["as_expected"],
                                                      "volta_s": x["volta_s"], "build_s": x["build_s"]}
                                                     for x in r["runs"]]} for r in d["rows"]}}
    return out


def run_perf(workloads: list[str]) -> dict:
    from akc.perf import measure
    from akc.validation.interface import Candidate
    out = {}
    for op in workloads:
        wl = load_workload(op)
        for v in wl.variants:
            if v.lang != "tirx" or v.bug:
                continue
            r = measure(Candidate.from_file(op, v.file, block=v.block, label=v.name))
            out[v.name] = {**r.to_dict(), "speedup_vs_torch": r.speedup_vs_torch}
            print(f"[perf] {v.name:24s} " + (f"{r.time_ms:.4f} ms, torch {r.torch_ms:.4f} ms, "
                                             f"x{r.speedup_vs_torch:.3f} torch, {r.achieved_gbps:.0f} GB/s, "
                                             f"{r.achieved_tflops:.3f} TFLOP/s" if r.measured else
                                             f"not measured: {r.reason}"), flush=True)
    d = RESULTS / "perf"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"tirx_task1_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"reports": out}, indent=1, default=str))
    print(f"[perf] saved {f}")
    return {"file": str(f.relative_to(ROOT)), "reports": out}


def summarize(res: dict, workloads: list[str]) -> dict:
    rows = {}
    for op, v in kernels(workloads):
        g = (res.get("gpu") or {}).get(v.name, {}).get("summary")
        n = (res.get("numsim") or {}).get(v.name, {}).get("summary")
        vk = ((res.get("volta") or {}).get(op) or {}).get("kernels", {}).get(v.name)
        caught = {"gpu": bool(g and g["failed"]), "numsim": bool(n and n["failed"]),
                  "volta": bool(vk and vk["caught"])}
        rows[v.name] = {"workload": op, "role": v.role, "bug": (v.bug or {}).get("category"),
                        "gpu": f"{g['passed']}/{g['total']} pass" + (f" ({','.join(g['reasons'])})" if g["failed"] else "")
                        if g else None,
                        "numsim": f"{n['passed']}/{n['total']} pass" + (f" ({','.join(n['reasons'])})" if n["failed"] else "")
                        if n else None,
                        "volta": {",".join(f"{k}={x}" for k, x in r["shape"].items()): r["status"]
                                  for r in vk["runs"]} if vk else None,
                        "volta_as_expected": all(r["as_expected"] is not False for r in vk["runs"]) if vk else None,
                        "caught_by": [k for k, x in caught.items() if x],
                        "ok": (not any(caught.values())) if not v.bug else any(caught.values())}
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="gpu,numsim,volta,perf")
    ap.add_argument("--workloads", default="layernorm,gemm")
    ap.add_argument("--gpu-workers", type=int, default=4)
    ap.add_argument("--numsim-workers", type=int, default=3)
    a = ap.parse_args(argv)
    parts, workloads = set(a.parts.split(",")), a.workloads.split(",")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ks = kernels(workloads)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as bg:                   # CPU-only parts in the background
        f_num = bg.submit(run_numsim, ks, a.numsim_workers) if "numsim" in parts else None
        f_vol = bg.submit(run_volta, workloads) if "volta" in parts else None
        if "gpu" in parts:
            res.setdefault("gpu", {}).update(run_gpu(ks, a.gpu_workers))
        if "perf" in parts:
            p = run_perf(workloads)
            res["perf"] = {"file": p["file"], "reports": {**(res.get("perf") or {}).get("reports", {}), **p["reports"]}}
        if f_num:
            res.setdefault("numsim", {}).update(f_num.result())
        if f_vol:
            res.setdefault("volta", {}).update(f_vol.result())
    res["date"] = datetime.now().isoformat(timespec="seconds")
    res["seconds_last_run"] = round(time.perf_counter() - t0, 1)
    done = [w for w in ("layernorm", "gemm") if w in workloads or any(k.startswith(w) for k in res.get("gpu", {}))]
    res["summary"] = summarize(res, done)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["summary"], indent=1))
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
