"""Run TestCases on the GPU and check every output (used by fixed tests and change-aware tests).

run_tests_isolated(workload, lang, kernel_file, entry, block, tests) -> list[outcome dict]
make_launcher(workload, lang, kernel_file, entry, block) -> bind(d_in, out, shape) -> launch()   (also used by perf)
  lang "cuda": kernel_file is a .ptx (built with backend.ptx mode="gpu"), launched through the driver API
  lang "tirx": kernel_file is a tirx-lite module (safety-scanned), compiled per shape for sm_80
Each candidate runs in a child process (`python -m akc.validation.runners.gpu_worker`): an illegal memory
access poisons the CUDA context, so the child stops at the first CUDA error, marks the remaining tests
"skipped", and the parent survives to test the next candidate.

Per test: inputs (seeded) -> float64 golden (PyTorch, on the GPU) -> launch `repeat` times into a sentinel-
filled buffer with canaries (check.alloc_output) -> compare run 0 with the golden -> bitwise-compare repeats
("nondeterministic" = race symptom). `misalign` shifts every tensor pointer by that many elements.
Outcome dict = TestCase fields + CheckResult fields + {"kernel_ms"}.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ...paths import BUILD, ROOT
from .check import CheckResult, alloc_output, compare


def _shifted(t, k: int):
    """Copy of tensor t whose data pointer is k elements past an aligned allocation."""
    import torch
    if not k:
        return t
    buf = torch.empty(t.numel() + k, dtype=t.dtype, device=t.device)
    v = buf[k:].view(t.shape)
    v.copy_(t)
    return v


def _run_one(launch, op, tc, out_shape) -> dict:
    import torch
    inputs = op.make_inputs(tc.shape, tc.dist, tc.seed)
    d_in = {k: _shifted(v.cuda(), tc.misalign) for k, v in inputs.items()}
    golden = op.golden({k: v.cuda() for k, v in inputs.items()})
    scale = op.magnitude({k: v.cuda() for k, v in inputs.items()}) if op.magnitude else None
    numel = golden.numel()
    outs, first_ms = [], None
    for _ in range(max(1, tc.repeat)):
        buf, out = alloc_output(numel, offset=tc.misalign)
        t0 = time.perf_counter()
        launch(d_in, out, tc)
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


def make_launcher(workload_name: str, lang: str, kernel_file: str, entry: str | None, block: tuple[int, ...]):
    """-> bind(d_in, out, shape) -> zero-argument function that launches the kernel once on those tensors.
    Binding (grid, arguments, TIRx compile per shape) happens outside the returned call, so timing it
    (perf.measure) measures the launch only. Needs a GPU."""
    from ...task.reference import OPS
    from ...task.workload import load_workload
    op = OPS[workload_name]
    wl = load_workload(workload_name)

    if lang == "cuda":
        from ...backend.launch import CudaModule
        fn = CudaModule(kernel_file).get(entry)

        def bind(d_in, out, shape):
            grid, blk = wl.launch_dims(shape, block)
            args = op.kernel_args(d_in, out, shape)
            return lambda: fn(grid, blk, args)
        return bind

    from ...backend.tirx import compile_sm80, load_module, make_kernel
    mod, cache, names = load_module(kernel_file), {}, wl.meta["tirx"]["tensors"]

    def bind(d_in, out, shape):
        key = tuple(sorted(shape.items()))
        if key not in cache:
            cache[key] = compile_sm80(make_kernel(mod, wl.meta, shape))
        ex = cache[key]
        args = [d_in[n] if n in d_in else out.view(op.out_shape(shape)) for n in names]
        return lambda: ex(*args)
    return bind


def run_tests(workload_name: str, lang: str, kernel_file: str, entry: str | None, block: tuple[int, ...],
              tests: list) -> list[dict]:
    """In-process (needs a GPU). Prefer run_tests_isolated."""
    from ...task.reference import OPS
    op = OPS[workload_name]
    bind = make_launcher(workload_name, lang, kernel_file, entry, block)

    def launch(d_in, out, tc):
        bind(d_in, out, tc.shape)()

    results, dead = [], None
    for tc in tests:
        if dead:
            results.append({**tc.to_dict(), **CheckResult(False, "skipped", message=f"skipped after: {dead}").to_dict()})
            continue
        try:
            results.append(_run_one(launch, op, tc, op.out_shape(tc.shape)))
        except Exception as e:  # noqa: BLE001  (CUDA faults surface as RuntimeError/CudaError; TIRx as others)
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            fatal = "illegal" in msg.lower() or "cuda" in type(e).__name__.lower() or "launch" in msg.lower()
            reason = "cuda_error" if fatal else "compile_error"
            results.append({**tc.to_dict(), **CheckResult(False, reason, message=msg).to_dict()})
            if fatal:
                dead = msg
    return results


def run_tests_isolated(workload_name: str, lang: str, kernel_file: str | Path, entry: str | None,
                       block: tuple[int, ...], tests: list, timeout_s: float = 180.0) -> list[dict]:
    (BUILD / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD / "tmp") as tmp:
        job, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job.write_text(json.dumps({"workload": workload_name, "lang": lang, "kernel_file": str(kernel_file),
                                   "entry": entry, "block": list(block), "tests": [t.to_dict() for t in tests]}))
        cmd = [sys.executable, "-m", "akc.validation.runners.gpu_worker", str(job), str(out)]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [{**t.to_dict(), **CheckResult(False, "cuda_error", message=f"timeout > {timeout_s}s (hang?)").to_dict()}
                    for t in tests]
        if out.exists():
            return json.loads(out.read_text())
        msg = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["worker died"]
        return [{**t.to_dict(), **CheckResult(False, "cuda_error", message=f"worker crashed: {msg[0]}").to_dict()}
                for t in tests]
