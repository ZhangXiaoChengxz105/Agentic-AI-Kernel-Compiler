"""Evaluate an LLM-written tirx-lite softmax module on the CPU (no GPU): safety scan, trace, sm_80
compile, NumSim execution vs a float64 reference, and Racecheck.

Contract for the module under test:
    def make_kernel(rows: int, n: int) -> txl.Kernel     # static shapes (NumSim needs them)
    kernel signature: (x: txl.Tensor((rows, n), "float32"), y: txl.Tensor((rows, n), "float32"))

safety_scan(src) -> list[str]        reasons the code must NOT be executed (empty list = ok)
evaluate_isolated(path, cases)       runs evaluate() in a child process with a timeout -> dict
  cases: [{"rows": 2, "n": 200, "dist": "normal"}, ...]
  result: {"stage": "safety|import|trace|compile|numsim|ok", "error": str, "cases": [...]}
Run from the workspace with RUSTUP_TOOLCHAIN=1.95.0 (NumSim builds its Rust engine on first use).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
ALLOWED_IMPORTS = {"tirx_kernels.tirx_lite", "tirx_kernels", "math", "typing", "functools"}
FORBIDDEN_NAMES = {"os", "sys", "subprocess", "shutil", "socket", "open", "eval", "exec", "compile", "__import__",
                   "importlib", "pathlib", "builtins", "globals", "locals", "setattr", "delattr", "input", "breakpoint"}


def safety_scan(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f"import of {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS:
                problems.append(f"import from {node.module} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute {node.attr} not allowed")
    return problems


def _softmax64(x):
    import numpy as np
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _inputs(rows, n, dist, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, n))
    if dist == "large":
        x = x * 100.0
    return x.astype(np.float32)


def evaluate(path: str, cases: list[dict]) -> dict:
    import importlib.util

    import numpy as np
    out = {"stage": "import", "error": "", "cases": []}
    try:
        spec = importlib.util.spec_from_file_location("candidate_txl", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        make_kernel = mod.make_kernel
    except Exception:  # noqa: BLE001  (report anything the candidate does wrong)
        out["error"] = traceback.format_exc(limit=3)[-2000:]
        return out

    from tirx_harness import numsim
    from tirx_harness.numsim.checkers import racecheck
    from tirx_kernels.runner import compile_kernel

    kernels = {}
    for c in cases:
        key = (c["rows"], c["n"])
        rec = {**c, "stage": "trace"}
        try:
            if key not in kernels:
                k = make_kernel(*key)
                rec["stage"] = "compile"
                compile_kernel(k.func, backend_config={"cuda": {"arch": "sm_80"}})
                rec["stage"] = "numsim"
                kernels[key] = (k, numsim.transpile(k.func))
            k, compiled = kernels[key]
            rec["stage"] = "numsim"
            x = _inputs(*key, c["dist"])
            res = numsim.Engine().run(compiled, inputs={"x": x, "y": np.zeros_like(x)})
            y = np.asarray(res.outputs["y"], dtype=np.float64)
            want = _softmax64(x)
            finite = np.isfinite(y)
            err = np.abs(y - want)
            rec.update(numsim_verdict=str(res.verdict), nonfinite=int((~finite).sum()),
                       max_abs_err=float(err[finite].max()) if finite.any() else float("inf"))
            bad = ~(err <= 1e-6 + 1e-4 * np.abs(want))
            rec["n_bad"] = int(bad.sum())
            if bad.any():
                i = tuple(int(v) for v in np.argwhere(bad)[0])
                rec["example"] = {"index": i, "got": float(y[i]), "want": float(want[i])}
            if c["dist"] == "normal":
                rc = racecheck(k.func, inputs={"x": x, "y": np.zeros_like(x)}).native_payload
                rec["racecheck"] = rc["verdict"]
                rec["race_findings"] = [str(f)[:400] for f in rc.get("findings", [])[:2]]
            rec["stage"] = "ok" if (rec["n_bad"] == 0 and rec.get("racecheck", "clean") == "clean") else "wrong"
        except Exception:  # noqa: BLE001
            rec["error"] = traceback.format_exc(limit=4)[-2500:]
        out["cases"].append(rec)
    stages = [r["stage"] for r in out["cases"]]
    out["stage"] = "ok" if all(s == "ok" for s in stages) else next(s for s in stages if s != "ok")
    return out


def evaluate_isolated(path: str | Path, cases: list[dict], timeout_s: float = 1800) -> dict:
    src = Path(path).read_text()
    problems = safety_scan(src)
    if problems:
        return {"stage": "safety", "error": "; ".join(problems), "cases": []}
    with tempfile.TemporaryDirectory(dir=WORKSPACE / "build") as tmp:
        res_file = Path(tmp) / "res.json"
        env = {**os.environ, "RUSTUP_TOOLCHAIN": "1.95.0"}
        cmd = [sys.executable, "-m", "agent.tirx_eval", str(path), json.dumps(cases), str(res_file)]
        try:
            p = subprocess.run(cmd, cwd=WORKSPACE, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"stage": "timeout", "error": f"evaluation exceeded {timeout_s}s", "cases": []}
        if res_file.exists():
            return json.loads(res_file.read_text())
        return {"stage": "crash", "error": (p.stderr or p.stdout)[-2000:], "cases": []}


if __name__ == "__main__":
    r = evaluate(sys.argv[1], json.loads(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(r))
