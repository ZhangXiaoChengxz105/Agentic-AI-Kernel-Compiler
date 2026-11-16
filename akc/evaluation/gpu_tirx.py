"""GPU check of tirx-lite softmax modules (hand-written reference + GLM outputs): correctness + speed.

    akc tirx-gpu [LABEL=PATH ...]        (GPU node; default: workload reference + GLM pilot outputs)
Correctness uses the shared GPU runner (runners.gpu, lang "tirx": safety scan, isolation, sentinel/canary
checks); timing runs in a child process at rows=4096, n=1024 against torch.softmax (L2 flushed, median of 50).
Output: table + results/harness/tirx_gpu_<ts>.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

from ..paths import RESULTS, ROOT
from ..task.testcase import TestCase
from ..task.workload import load_workload

PILOT = {
    "glm_C (r2)": "results/pilot/20261115-203843/softmax_txl_r2.py",
    "glm_D1 (r2)": "results/pilot/20261115-204306/softmax_txl_r2.py",
    "glm_D2 (r1)": "results/pilot/20261115-205635/softmax_txl_r1.py",
    "glm_D3 (r1)": "results/pilot/20261115-205831/softmax_txl_r1.py",
}
TESTS = [TestCase("softmax", {"rows": r, "N": n}, d, seed=i, tag="tirx-gpu")
         for i, (r, n, d) in enumerate([(64, 256, "normal"), (64, 1000, "normal"), (64, 4096, "normal"), (3, 129, "normal"),
                                        (2, 1, "normal"), (8, 1000, "large"), (8, 1000, "neg_large"), (8, 1000, "constant")])]


def _bench(path: str) -> dict:
    import torch

    from ..backend.bench import bench
    from ..backend.tirx import compile_sm80, load_module, make_kernel
    wl = load_workload("softmax")
    ex = compile_sm80(make_kernel(load_module(path), wl.meta, {"rows": 4096, "N": 1024}))
    x = torch.randn(4096, 1024, device="cuda")
    y = torch.empty_like(x)
    return {"kernel": bench(lambda: ex(x, y)), "torch": bench(lambda: torch.softmax(x, dim=-1))}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--bench":
        print("RESULT_JSON " + json.dumps(_bench(argv[1])))
        return 0
    from ..validation.runners.gpu import run_tests_isolated
    wl = load_workload("softmax")
    targets = dict(x.split("=", 1) for x in argv) if argv else {"txl_ref": str(wl.reference("tirx").file), **PILOT}
    results = {}
    for label, path in targets.items():
        outcomes = run_tests_isolated("softmax", "tirx", str(ROOT / path), None, (128,), TESTS)
        p = subprocess.run([sys.executable, "-m", "akc.evaluation.gpu_tirx", "--bench", str(ROOT / path)], cwd=ROOT,
                           capture_output=True, text=True, timeout=1800)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
        b = json.loads(line[12:]) if line else {"error": (p.stderr or "")[-300:]}
        results[label] = {"outcomes": outcomes, "bench": b}
        npass = sum(o["passed"] for o in outcomes)
        speed = (f"{b['kernel']['median_ms']:.4f} ms (torch {b['torch']['median_ms']:.4f} ms, "
                 f"x{b['torch']['median_ms'] / b['kernel']['median_ms']:.2f})") if "kernel" in b else b.get("error", "-")
        print(f"{label:24s} {npass}/{len(outcomes)} tests passed   rows=4096,n=1024: {speed}")
        for o in outcomes:
            if not o["passed"]:
                print(f"     FAIL {o['shape']} {o['dist']}: {o['reason']} {o['message'][:100]}")
    d = RESULTS / "harness"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"tirx_gpu_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(results, indent=2, default=str))
    print(f"saved {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
