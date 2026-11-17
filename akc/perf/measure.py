"""Measure one candidate's speed on the GPU -> PerfReport.

measure(cand, shape=None, *, iters=50, warmup=10, check=True, device="auto") -> PerfReport
  1. shape: the manifest's "perf.shape" unless given (softmax/layernorm rows=4096,N=1024; gemm 1024^3);
     a workload without perf.shape or without OpSpec flops/bytes/torch_impl -> measured=False, "not set up";
     a shape outside the manifest's declared "domain" -> measured=False (speed counts only where correctness
     is checked: the CAKE contract)
  2. no GPU here -> PerfReport(measured=False, reason="no GPU ...") (VOLTA runs on the CPU, speed cannot)
  3. build: CUDA -> PTX (same build as the GPU tests); compile error -> measured=False
  4. child process (like the GPU test runner: a crashing kernel must not take the caller down):
       a. correctness at the benchmark shape (one N(0,1) test, same checks as the tests). A fast wrong
          kernel is meaningless, and the validated sizes may differ from the benchmark size.
       b. kernel time: backend.bench (warmup, CUDA events, L2 flushed before every launch, median)
       c. PyTorch's implementation of the op under the same conditions (speed baseline)
       d. a device copy moving the op's minimum bytes (practical floor for memory-bound ops)
  5. perf.report.analyze -> roofline numbers (bandwidth / FLOP/s achieved vs the GPU's peaks)
Only call this for candidates that passed validation; configurations A/B/C must measure identically.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..paths import BUILD, ROOT
from .hardware import spec_for
from .report import PerfReport, analyze


def measure(cand, shape: dict | None = None, *, iters: int = 50, warmup: int = 10, check: bool = True,
            device: str = "auto", timeout_s: float = 900.0) -> PerfReport:
    from ..task.reference import OPS
    from ..task.workload import load_workload
    from ..validation.interface import ValidationContext
    from ..validation.runners import BuildError, build_for_gpu
    t0 = time.perf_counter()
    wl = load_workload(cand.workload)
    shape = dict(shape or wl.perf_shape or {})
    skip = lambda why: PerfReport.not_measured(cand.workload, cand.label, cand.lang, shape, why,  # noqa: E731
                                               time.perf_counter() - t0)
    if not shape or not OPS[cand.workload].has_perf:
        return skip("speed is not set up for this workload (needs perf.shape in the manifest and "
                    "flops/bytes/torch_impl in its OpSpec)")
    if not wl.in_domain(shape):
        return skip(f"shape {shape} is outside the declared shape domain ({wl.domain_text()}): correctness is "
                    "only checked inside it, so speed is not measured outside it")
    if ValidationContext(device=device).resolved_device() != "gpu":
        return skip("no GPU here: speed can only be measured on a GPU node")
    try:
        kernel_file, entry, block = build_for_gpu(cand)
    except BuildError as e:
        return skip(f"compile error: {str(e)[-400:]}")
    job = {"workload": cand.workload, "lang": cand.lang, "kernel_file": str(kernel_file), "entry": entry,
           "block": list(block), "shape": shape, "iters": iters, "warmup": warmup, "check": check}
    (BUILD / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD / "tmp") as tmp:
        jf, of = Path(tmp) / "job.json", Path(tmp) / "out.json"
        jf.write_text(json.dumps(job))
        try:
            proc = subprocess.run([sys.executable, "-m", "akc.perf.worker", str(jf), str(of)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return skip(f"timeout > {timeout_s:.0f}s while measuring (hang?)")
        if not of.exists():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["worker died"]
            return skip(f"measurement crashed: {tail[0][:300]}")
        res = json.loads(of.read_text())
    if res.get("error"):
        return skip(f"runtime error at the benchmark shape: {res['error']}")
    chk = res.get("check")
    if chk and not chk["passed"]:
        return skip(f"incorrect at the benchmark shape {shape}: {chk['reason']} ({chk.get('message', '')[:200]})")
    return analyze(cand.workload, shape, res["timing"], spec_for(res.get("gpu")), label=cand.label, lang=cand.lang,
                   torch_ms=res["torch_ms"], copy_ms=res["copy_ms"], seconds=time.perf_counter() - t0)


def _child(job: dict) -> dict:
    """Runs in the child process (needs a GPU)."""
    import torch

    from ..backend.bench import bench
    from ..task.reference import OPS
    from ..task.testcase import TestCase
    from ..validation.runners.check import alloc_output
    from ..validation.runners.gpu import _run_one, make_launcher
    w, shape, iters, warmup = job["workload"], job["shape"], job["iters"], job["warmup"]
    op = OPS[w]
    res = {"gpu": torch.cuda.get_device_name()}
    try:
        bind = make_launcher(w, job["lang"], job["kernel_file"], job["entry"], tuple(job["block"]))
        if job["check"]:
            tc = TestCase(w, shape, "normal", seed=0, repeat=1, tag="perf: correctness at the benchmark shape")
            res["check"] = _run_one(lambda d, o, t: bind(d, o, t.shape)(), op, tc, op.out_shape(shape))
            if not res["check"]["passed"]:
                return res
        inputs = {k: v.cuda() for k, v in op.make_inputs(shape, "normal", 0).items()}
        _, out = alloc_output(int(torch.Size(op.out_shape(shape)).numel()))
        res["timing"] = bench(bind(inputs, out, shape), warmup, iters)
        res["torch_ms"] = bench(lambda: op.torch_impl(inputs), warmup, iters)["median_ms"]
        n = max(1, int(op.bytes(shape) // 8))           # a float copy reads 4 and writes 4 bytes per element
        src = torch.empty(n, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        res["copy_ms"] = bench(lambda: dst.copy_(src), warmup, iters)["median_ms"]
    except Exception as e:  # noqa: BLE001  (CUDA faults, TIRx compile errors)
        res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    return res
