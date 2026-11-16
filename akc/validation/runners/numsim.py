"""Run TestCases for tirx-lite candidates on the CPU with TIRx NumSim (+ Racecheck): no GPU needed.

run_tests_numsim_isolated(workload, module_path, tests) -> list[outcome dict]   (same format as runners.gpu)

The module is safety-scanned, then imported in a child process (`python -m akc.validation.runners.numsim`),
with RUSTUP_TOOLCHAIN=1.95.0 (NumSim builds its Rust engine; cargo 1.100's new build-dir layout breaks
tirx-harness's artifact lookup) and CARGO_BUILD_JOBS=4 (be gentle on the login node).
Costs: every distinct shape is transpiled to a native artifact (~0.5-1 min first time, cached), so the runner
caps work: at most `max_shapes` distinct shapes, rows <= `max_rows`, N <= `max_n`; other tests come back
"skipped" (an info finding, never a pass). Racecheck runs on the first shape and on tests tagged "sync".
The output buffer is sentinel-filled before simulation, so unwritten elements are still detected; NumSim
itself reports out-of-bounds accesses (exact byte range), which become "out_of_bounds" outcomes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from ...backend.tirx import safety_scan
from ...paths import BUILD, ROOT
from .check import CheckResult, alloc_output, compare


def _evaluate(workload_name: str, module_path: str, tests: list, max_shapes: int, max_rows: int, max_n: int) -> list:
    import numpy as np
    import torch

    from tirx_harness import numsim
    from tirx_harness.numsim.checkers import racecheck

    from ...backend.tirx import compile_sm80, load_module, make_kernel
    from ...task.reference import OPS
    from ...task.workload import load_workload
    op, wl = OPS[workload_name], load_workload(workload_name)
    names = wl.meta["tirx"]["tensors"]
    mod = load_module(module_path)
    cache, raced, out = {}, set(), []
    for i, tc in enumerate(tests):
        key = tuple(sorted(tc.shape.items()))
        too_big = tc.shape.get("rows", 1) > max_rows or tc.shape.get("N", 0) > max_n
        if too_big or (key not in cache and len(cache) >= max_shapes):
            why = "shape exceeds NumSim caps" if too_big else f"more than {max_shapes} distinct shapes"
            out.append({**tc.to_dict(), **CheckResult(False, "skipped", message=f"not simulated on CPU: {why}").to_dict()})
            continue
        t0 = time.perf_counter()
        stage = "compile"
        try:
            if key not in cache:
                k = make_kernel(mod, wl.meta, tc.shape)
                compile_sm80(k)                                    # same compile path as the GPU
                stage = "numsim"
                cache[key] = (k, numsim.transpile(k.func))
            k, compiled = cache[key]
            stage = "numsim"
            inputs = op.make_inputs(tc.shape, tc.dist, tc.seed)
            buf, y = alloc_output(int(np.prod(op.out_shape(tc.shape))), device="cpu")
            arrays = {n: (inputs[n].numpy() if n in inputs else y.view(op.out_shape(tc.shape)).numpy()) for n in names}
            res = numsim.Engine(max_workers=2).run(compiled, inputs=arrays)
            out_name = names[-1]
            y.copy_(torch.from_numpy(np.asarray(res.outputs[out_name], dtype=np.float32)).flatten())
            r = compare(buf, y, op.golden(inputs), atol=op.atol, rtol=op.rtol,
                        scale=op.magnitude(inputs) if op.magnitude else None)
            rec = {**tc.to_dict(), **r.to_dict(), "kernel_ms": None, "sim_seconds": time.perf_counter() - t0}
            if (i == 0 or tc.tag.startswith("sync")) and key not in raced:
                raced.add(key)
                arrays2 = {n: (inputs[n].numpy() if n in inputs else np.zeros(op.out_shape(tc.shape), np.float32)) for n in names}
                rc = racecheck(k.func, inputs=arrays2).native_payload
                if rc["verdict"] != "clean":
                    finding = str((rc.get("findings") or [""])[0])[:600]
                    rec.update(passed=False, reason="race", message=f"Racecheck: data race ({finding[:200]})")
            out.append(rec)
        except Exception as e:  # noqa: BLE001
            msg = traceback.format_exc(limit=2)[-1500:]
            reason = "out_of_bounds" if "out-of-bounds" in str(e) else ("compile_error" if stage == "compile" else "runtime_error")
            out.append({**tc.to_dict(), **CheckResult(False, reason, message=f"{type(e).__name__}: {str(e)[:400]}").to_dict(),
                        "traceback": msg})
    return out


def run_tests_numsim_isolated(workload_name: str, module_path: str | Path, tests: list, max_shapes: int = 4,
                              max_rows: int = 4, max_n: int = 2048, timeout_s: float = 2400.0) -> list[dict]:
    problems = safety_scan(Path(module_path).read_text())
    if problems:
        return [{**t.to_dict(), **CheckResult(False, "compile_error", message="refused to import: " + "; ".join(problems)).to_dict()}
                for t in tests]
    (BUILD / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD / "tmp") as tmp:
        job, res_file = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job.write_text(json.dumps({"workload": workload_name, "module": str(module_path), "tests": [t.to_dict() for t in tests],
                                   "max_shapes": max_shapes, "max_rows": max_rows, "max_n": max_n}))
        env = {**os.environ, "RUSTUP_TOOLCHAIN": "1.95.0", "CARGO_BUILD_JOBS": "4"}
        cmd = [sys.executable, "-m", "akc.validation.runners.numsim", str(job), str(res_file)]
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [{**t.to_dict(), **CheckResult(False, "skipped", message=f"NumSim timeout > {timeout_s}s").to_dict()}
                    for t in tests]
        if res_file.exists():
            return json.loads(res_file.read_text())
        err = (p.stderr or p.stdout)[-1500:]
        return [{**t.to_dict(), **CheckResult(False, "compile_error", message=f"NumSim worker failed: {err}").to_dict()}
                for t in tests]


if __name__ == "__main__":
    from ...task.testcase import TestCase
    j = json.loads(Path(sys.argv[1]).read_text())
    res = _evaluate(j["workload"], j["module"], [TestCase.from_dict(t) for t in j["tests"]],
                    j["max_shapes"], j["max_rows"], j["max_n"])
    Path(sys.argv[2]).write_text(json.dumps(res, default=str))
