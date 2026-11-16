#!/usr/bin/env python3
"""GPU check of tirx-lite softmax modules (hand-written reference + GLM outputs): correctness + speed.

Each module (contract: make_kernel(rows, n) -> txl.Kernel with x, y float32 [rows, n]) runs in its own child
process (a CUDA fault cannot poison the others), after the same safety scan used for NumSim evaluation.
Per module: compile for sm_80 per shape, run on the GPU, compare with harness.check.compare (sentinel/canary,
fp32 tolerances vs float64 torch.softmax), then time rows=4096, n=1024 with harness.bench (L2 flushed).

Usage (GPU node): python scripts/gpu_tirx_check.py [LABEL=PATH ...]     (default: reference + GLM pilot outputs)
Output: table + results/harness/tirx_gpu_<ts>.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

DEFAULT = {
    "txl_ref (hand-written)": "kernels_txl/softmax/softmax_txl_ref.py",
    "glm_C (r2)": "results/pilot/20261115-203843/softmax_txl_r2.py",
    "glm_D1 (r2)": "results/pilot/20261115-204306/softmax_txl_r2.py",
    "glm_D2 (r1)": "results/pilot/20261115-205635/softmax_txl_r1.py",
    "glm_D3 (r1)": "results/pilot/20261115-205831/softmax_txl_r1.py",
}
CASES = [(64, 256, "normal"), (64, 1000, "normal"), (64, 4096, "normal"), (3, 129, "normal"), (2, 1, "normal"),
         (8, 1000, "large"), (8, 1000, "neg_large"), (8, 1000, "constant")]
BENCH = (4096, 1024)


def run_one(path: str) -> dict:
    import importlib.util

    import torch

    from agent.tirx_eval import safety_scan
    from harness.bench import bench
    from harness.check import alloc_output, compare
    from ref.ops import OPS
    from tirx_kernels.runner import compile_kernel

    problems = safety_scan(Path(path).read_text())
    if problems:
        return {"error": "safety: " + "; ".join(problems)}
    spec = importlib.util.spec_from_file_location("cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    op = OPS["softmax"]
    out = {"cases": []}
    for rows, n, dist in CASES:
        shape = {"rows": rows, "N": n}
        rec = {"rows": rows, "n": n, "dist": dist}
        try:
            ex = compile_kernel(mod.make_kernel(rows, n).func, backend_config={"cuda": {"arch": "sm_80"}})
            inputs = op.make_inputs(shape, dist, 0)
            x = inputs["x"].cuda()
            buf, y = alloc_output(rows * n)
            ex(x, y.view(rows, n))
            torch.cuda.synchronize()
            r = compare(buf, y, op.golden(inputs), atol=op.atol, rtol=op.rtol)
            rec.update(passed=r.passed, reason=r.reason, max_abs_err=r.max_abs_err, message=r.message)
        except Exception as e:  # noqa: BLE001
            rec.update(passed=False, reason="error", message=f"{type(e).__name__}: {str(e)[:300]}")
            out["cases"].append(rec)
            if "illegal" in str(e).lower():
                break
            continue
        out["cases"].append(rec)
    rows, n = BENCH
    try:
        ex = compile_kernel(mod.make_kernel(rows, n).func, backend_config={"cuda": {"arch": "sm_80"}})
        x = torch.randn(rows, n, device="cuda")
        y = torch.empty_like(x)
        out["bench"] = bench(lambda: ex(x, y))
        out["torch"] = bench(lambda: torch.softmax(x, dim=-1))
    except Exception as e:  # noqa: BLE001
        out["bench_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print("RESULT_JSON " + json.dumps(run_one(sys.argv[2])))
        return 0
    targets = dict(a.split("=", 1) for a in sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT
    results = {}
    for label, path in targets.items():
        p = subprocess.run([sys.executable, __file__, "--one", path], cwd=WORKSPACE, capture_output=True, text=True,
                           timeout=1800)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
        results[label] = json.loads(line[12:]) if line else {"error": (p.stderr or p.stdout)[-800:]}
        r = results[label]
        if "error" in r:
            print(f"{label:24s} ERROR {r['error'][:200]}")
            continue
        npass = sum(c["passed"] for c in r["cases"])
        b = r.get("bench", {}).get("median_ms")
        t = r.get("torch", {}).get("median_ms")
        speed = f"{b:.4f} ms (torch {t:.4f} ms, x{t / b:.2f})" if b and t else r.get("bench_error", "-")
        print(f"{label:24s} {npass}/{len(r['cases'])} cases passed   bench rows=4096,n=1024: {speed}")
        for c in r["cases"]:
            if not c["passed"]:
                print(f"     FAIL rows={c['rows']} n={c['n']} {c['dist']}: {c['reason']} {c['message'][:100]}")
    d = WORKSPACE / "results" / "harness"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"tirx_gpu_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(results, indent=2))
    print(f"saved {f.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
