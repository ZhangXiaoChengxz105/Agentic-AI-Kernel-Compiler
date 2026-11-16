"""Task 1.5: run a list of TestCases against one compiled kernel and check every result.

run_tests(op, ptx, kernel, block, tests) -> list[dict]          (in this process, needs a GPU)
run_tests_isolated(op, ptx, kernel, block, tests) -> list[dict] (in a child process)

Why isolation: an out-of-bounds access can raise "illegal memory access", which poisons the CUDA
context for the rest of the process. The child stops at the first CUDA error, marks the remaining
tests "skipped", and the parent survives to test the next kernel.

Each result dict = TestCase fields + CheckResult fields + {"kernel_ms": first-launch time}.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from ref.ops import OPS
from .check import CheckResult, alloc_output, compare
from .launch import CudaError, CudaModule
from .testcase import TestCase

WORKSPACE = Path(__file__).resolve().parents[1]


def run_one(fn, op_name: str, block: tuple[int, ...], tc: TestCase) -> dict:
    op = OPS[op_name]
    inputs = op.make_inputs(tc.shape, tc.dist, tc.seed)
    d_in = {k: v.cuda() for k, v in inputs.items()}
    golden = op.golden({k: v.cuda() for k, v in inputs.items()})       # float64 on GPU (fast for gemm)
    scale = op.magnitude({k: v.cuda() for k, v in inputs.items()}) if op.magnitude else None
    numel = golden.numel()
    grid, blk = op.launch(tc.shape, block)

    outs, first_ms = [], None
    for _ in range(max(1, tc.repeat)):
        buf, out = alloc_output(numel)
        t0 = time.perf_counter()
        fn(grid, blk, op.kernel_args(d_in, out, tc.shape))
        torch.cuda.synchronize()
        first_ms = first_ms if first_ms is not None else (time.perf_counter() - t0) * 1e3
        outs.append((buf, out))

    res = compare(outs[0][0], outs[0][1], golden, atol=op.atol, rtol=op.rtol, scale=scale)
    if res.passed and len(outs) > 1:
        ref_bits = outs[0][1].view(torch.int32)
        differing = [i for i, (_, o) in enumerate(outs[1:], 1) if not torch.equal(o.view(torch.int32), ref_bits)]
        if differing:
            res.passed, res.reason = False, "nondeterministic"
            res.message = (f"output differs between identical launches ({len(differing)} of {len(outs) - 1} "
                           f"repeats differ from run 0): likely a data race")
    return {**tc.to_dict(), **res.to_dict(), "kernel_ms": first_ms}


def run_tests(op_name: str, ptx: str | Path, kernel: str, block: tuple[int, ...], tests: list[TestCase]) -> list[dict]:
    fn = CudaModule(ptx).get(kernel)
    results, dead = [], None
    for tc in tests:
        if dead:
            results.append({**tc.to_dict(), **CheckResult(False, "skipped", message=f"skipped after: {dead}").to_dict()})
            continue
        try:
            results.append(run_one(fn, op_name, block, tc))
        except CudaError as e:
            dead = str(e)
            results.append({**tc.to_dict(), **CheckResult(False, "cuda_error", message=dead).to_dict()})
        except RuntimeError as e:                      # torch surfaces async CUDA errors as RuntimeError
            dead = str(e).splitlines()[0]
            results.append({**tc.to_dict(), **CheckResult(False, "cuda_error", message=dead).to_dict()})
    return results


def run_tests_isolated(op_name: str, ptx: str | Path, kernel: str, block: tuple[int, ...],
                       tests: list[TestCase], timeout_s: float = 600.0) -> list[dict]:
    with tempfile.TemporaryDirectory(dir=WORKSPACE / "build") as tmp:
        spec = Path(tmp) / "job.json"
        out = Path(tmp) / "out.json"
        spec.write_text(json.dumps({"op": op_name, "ptx": str(ptx), "kernel": kernel, "block": list(block),
                                    "tests": [t.to_dict() for t in tests]}))
        cmd = [sys.executable, "-m", "harness.worker", str(spec), str(out)]
        try:
            proc = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [{**t.to_dict(), **CheckResult(False, "cuda_error", message=f"timeout > {timeout_s}s (hang?)").to_dict()}
                    for t in tests]
        if out.exists():
            return json.loads(out.read_text())
        msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["worker died"]
        return [{**t.to_dict(), **CheckResult(False, "cuda_error", message=f"worker crashed: {msg[0]}").to_dict()}
                for t in tests]
