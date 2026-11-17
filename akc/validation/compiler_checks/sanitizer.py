"""Dynamic compiler checks on a GPU: NVIDIA compute-sanitizer on the binary built from a tirx-lite kernel's generated
CUDA (the `.cu` files tirx_static writes, i.e. `Kernel.source()`).

    find_compute_sanitizer() -> path | None
    build_cubin(cu, out_dir) -> (cubin path, seconds)              nvcc -cubin -arch=sm_80 -lineinfo   (CompileError)
    write_inputs(workload, shape, params, out_dir, seed) -> (args, golden, op)   parent: tensors as .npy files
    run_tools(units, tools, out_dir, log_dir, cs, parallel=True) -> {tool: run dict}
    parse_sanitizer(text, entry) -> dict                           (pure; unit-tested on saved logs)
    report_kind(tool, head) -> Finding kind | None
Child (the process every tool wraps): launcher.py, run by path (it imports only numpy and cuda.bindings).

Tools (flags as in scripts/experiments/sanitizer_trial.py, where they gave 0 false alarms on 7 correct CUDA kernels,
except memcheck's error policy):
  memcheck   --padding 256 --destroy-on-device-error context    out-of-bounds / misaligned global, shared, local access.
                                                                The trial's "kernel" policy hung for the full 180 s
                                                                timeout on 2 GLM tirx mutants after the first report
                                                                (job 24061708); "context" ends the run at the first
                                                                error (later shapes of that run are skipped).
  racecheck  --racecheck-detect-level info                      shared-memory hazards (error / warning / info)
  initcheck  --initcheck-address-space all                      reads of uninitialized global AND shared memory; the
                                                                output buffers are left uninitialized in this run, so a
                                                                kernel that reads its output before writing it shows
  synccheck                                                     invalid / divergent barriers (-> "deadlock")
The four tools run in parallel, one process each (the trial ran 6 tool processes at once on one A100 without problems).
Each process launches the kernel at every gate shape (one cubin per shape: TIRx bakes the shape in). A report is
attributed to a shape by its source file name (each shape has its own .cu).

The launcher child imports only numpy and cuda.bindings (no torch: no PyTorch kernels, no torch start-up under the
tools). Every tensor is its own exact-size cuMemAlloc (memcheck's --padding then sees an overflow into the next
buffer), inputs come from .npy files the parent wrote (OpSpec.make_inputs), outputs are pre-filled with the harness
sentinel (except under initcheck) and saved after the memcheck run, so the parent can also check values.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...paths import ROOT

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
SANITIZER_FLAGS = {
    "memcheck": ["--tool", "memcheck", "--padding", "256", "--destroy-on-device-error", "context"],
    "racecheck": ["--tool", "racecheck", "--racecheck-detect-level", "info"],
    "initcheck": ["--tool", "initcheck", "--initcheck-address-space", "all"],
    "synccheck": ["--tool", "synccheck"],
}
COMMON_FLAGS = ["--show-backtrace", "device", "--print-limit", "30"]
DEFAULT_CS = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/compute-sanitizer/compute-sanitizer"
RUN_TIMEOUT_S = 60            # the gate shapes run in ~2 s under any tool; a hang must not cost minutes
SENTINEL_BITS = 0x7FC0DEAD
LAUNCHER = Path(__file__).resolve().with_name("launcher.py")      # run by path: no akc (torch) import in the child


def find_compute_sanitizer() -> str | None:
    """$AKC_COMPUTE_SANITIZER, compute-sanitizer on PATH, the HPC SDK default, or next to nvcc."""
    for c in (os.environ.get("AKC_COMPUTE_SANITIZER"), shutil.which("compute-sanitizer"), DEFAULT_CS):
        if c and Path(c).exists():
            return str(c)
    nvcc = shutil.which("nvcc")
    if nvcc:
        c = Path(nvcc).resolve().parent.parent / "compute-sanitizer" / "compute-sanitizer"
        if c.exists():
            return str(c)
    return None


def build_cubin(cu: str | Path, out_dir: str | Path | None = None, arch: str = "sm_80") -> tuple[Path, float]:
    """nvcc -cubin with line info (sanitizer reports then name lines of the generated CUDA)."""
    from ...backend.ptx import CompileError
    cu = Path(cu)
    out = Path(out_dir or cu.parent) / f"{cu.stem}.cubin"
    out.parent.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise FileNotFoundError("nvcc not found on PATH (module load cudatoolkit)")
    cmd = [nvcc, f"-arch={arch}", "-cubin", "-lineinfo", str(cu), "-o", str(out)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise CompileError(cmd, p.stderr)
    return out, time.perf_counter() - t0


def write_inputs(workload: str, shape: dict, params: list[str], out_dir: Path, seed: int = 0, dist: str = "normal"):
    """Inputs of OpSpec.make_inputs and sentinel-filled outputs as .npy files, in kernel-parameter order.
    Returns (args for the launcher spec, float64 golden, OpSpec)."""
    import numpy as np

    from ...task.reference import OPS
    op = OPS[workload]
    inputs = op.make_inputs(shape, dist, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = []
    for name in params:
        f = out_dir / f"{name}.npy"
        if name in inputs:
            np.save(f, np.ascontiguousarray(inputs[name].numpy().astype(np.float32)))
            args.append({"name": name, "npy": str(f), "output": False})
        else:
            n = int(np.prod(op.out_shape(shape)))
            np.save(f, np.full(n, SENTINEL_BITS, dtype=np.uint32).view(np.float32).reshape(op.out_shape(shape)))
            args.append({"name": name, "npy": str(f), "output": True, "save": str(out_dir / f"{name}.out.npy")})
    return args, op.golden(inputs), op, inputs


# ---- parent: run the tools ------------------------------------------------------------------------------------------
def run_tool(tool: str, units: list[dict], out_dir: Path, log: Path, cs: str, timeout_s: float = RUN_TIMEOUT_S) -> dict:
    """One compute-sanitizer process (`tool`) around one launcher child that runs every unit (shape)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    spec, out = out_dir / f"{tool}.spec.json", out_dir / f"{tool}.out.json"
    for p in (out, log):
        if p.exists():
            p.unlink()
    save_outputs = tool == "memcheck"
    spec.write_text(json.dumps({"units": units, "init_outputs": tool != "initcheck", "save_outputs": save_outputs}))
    child = [sys.executable, str(LAUNCHER), str(spec), str(out)]
    cmd = [cs, *SANITIZER_FLAGS[tool], *COMMON_FLAGS, "--log-file", str(log), *child]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
        rc, tail = p.returncode, (p.stderr or p.stdout).strip().splitlines()[-4:]
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", []
    secs = time.perf_counter() - t0
    child_res = json.loads(out.read_text()) if out.exists() else None
    text = log.read_text(errors="replace") if log.exists() else ""
    return {"tool": tool, "seconds": round(secs, 2), "exit": rc, "log": str(log), "child": child_res, "tail": tail,
            "text": text}


def run_tools(units: list[dict], tools=TOOLS, out_dir: Path | None = None, log_dir: Path | None = None,
              cs: str | None = None, parallel: bool = True, timeout_s: float = RUN_TIMEOUT_S) -> dict[str, dict]:
    cs = cs or find_compute_sanitizer()
    if cs is None:
        raise FileNotFoundError("compute-sanitizer not found (module load cudatoolkit/26.5_13.2)")

    def one(tool):
        return run_tool(tool, units, out_dir / tool, log_dir / f"{tool}.log", cs, timeout_s)
    if parallel and len(tools) > 1:
        with ThreadPoolExecutor(len(tools)) as ex:
            res = list(ex.map(one, tools))
    else:
        res = [one(t) for t in tools]
    return {r["tool"]: r for r in res}


def sass_line_map(cubin: str | Path, entry: str | None = None) -> dict[int, int]:
    """{SASS offset in the kernel: line of the generated CUDA in the kernel body}, from `nvdisasm -gi` (line info with
    inlining). TIRx emits every load / store / barrier through an inlined helper (tvm_builtin_ptx_st_shared_f32, ...),
    so sanitizer reports name the helper's line; its "inlined at" chain gives the call site in the kernel. The offsets
    are the ones compute-sanitizer prints as `<function>+0x<offset>` (checked on job 24061664: they land exactly on
    the STS / LDS / STG instructions of the reported accesses). Empty dict when nvdisasm is missing or fails."""
    nvdisasm = shutil.which("nvdisasm")
    if not nvdisasm and shutil.which("nvcc"):
        cand = Path(shutil.which("nvcc")).resolve().parent / "nvdisasm"
        nvdisasm = str(cand) if cand.exists() else None
    if not nvdisasm:
        return {}
    try:
        p = subprocess.run([nvdisasm, "-gi", str(cubin)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if p.returncode != 0:
        return {}
    out, in_text, cur = {}, False, None
    for line in p.stdout.splitlines():
        t = line.strip()
        sec = re.match(r"\.section\s+(\S+)", t)          # not .sectioninfo / .sectionflags
        if sec:
            in_text = sec.group(1).startswith(".text." + (entry or ""))
            continue
        if not in_text:
            continue
        if t.startswith("//## File"):
            nums = re.findall(r"line (\d+)", t)
            cur = int(nums[-1]) if nums else cur          # the outermost frame: the line in the kernel body
            continue
        m = re.match(r"/\*([0-9a-fA-F]{4,})\*/\s+(?!\.)", t)
        if m and cur is not None:
            out[int(m.group(1), 16)] = cur
    return out


# ---- parsing ----------------------------------------------------------------------------------------------------------
_SKIP_HEAD = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "LEAK SUMMARY", "Target application")
_LOC_RE = re.compile(r" in (\S+?\.cuh?):(\d+)")
_PC_RE = re.compile(r"(?:\b(?P<acc>Write|Read|Atomic) access at |\bat |Device Frame:\s*)(?P<fn>\S.*?)\+0x(?P<off>[0-9a-fA-F]+)"
                    r" in (?P<file>\S+?\.cuh?):(?P<line>\d+)")


def report_kind(tool: str, head: str) -> str | None:
    """Finding kind of one sanitizer report (None: not a kernel defect we count, e.g. initcheck's unused memory)."""
    h = head.lower()
    if tool == "racecheck" or "race reported" in h or "hazard" in h:
        return "race"
    if tool == "initcheck" or "uninitialized" in h:
        return None if "unused memory" in h else "uninitialized"
    if tool == "synccheck" or "barrier" in h or "syncwarp" in h:
        return "deadlock"
    if re.search(r"invalid|out of bounds|out-of-range|misaligned|address", h):
        return "out_of_bounds"
    return "runtime_error"


def parse_sanitizer(text: str, entry: str, tool: str | None = None, files=None) -> dict:
    """Split a compute-sanitizer log into report blocks; keep those that are ours: they name our kernel (`entry`) or,
    with `files` (basenames of our generated .cu files), a source line in one of them. TIRx's accesses sit in inlined
    helpers, so a racecheck / initcheck report names the helper (tvm_builtin_ptx_st_shared_f32(...)+0x260), not the
    kernel. Same block rules as the CUDA trial's parser (results/eval/sanitizer_trial.json), plus each report's kind,
    its source locations (file, line) and its PCs (access, function, offset, file, line) in order."""
    blocks, cur = [], None
    for line in text.splitlines():
        if not line.startswith("========="):
            continue
        body = line[len("========="):]
        if body.startswith(" ") and len(body) > 1 and not body[1].isspace():
            cur = [body.strip()]
            blocks.append(cur)
        elif cur is not None and body.strip():
            cur.append(body.strip())
    reports, api_errors = [], []
    for b in blocks:
        head = b[0]
        if head.startswith(_SKIP_HEAD):
            continue
        if head.startswith("Program hit"):
            api_errors.append(head[:200])
            continue
        txt = " ".join(b)
        m = re.match(r"(error|warning|info)\b", head, re.I)
        sev = m.group(1).lower() if m else "error"
        locs = []
        for f, ln in _LOC_RE.findall(txt):
            loc = (Path(f).name, int(ln))
            if loc not in locs:
                locs.append(loc)
        pcs = [{"access": (m.group("acc") or "").lower() or None, "fn": m.group("fn")[:80],
                "off": int(m.group("off"), 16), "file": Path(m.group("file")).name, "line": int(m.group("line"))}
               for m in _PC_RE.finditer(txt)]
        hz = re.findall(r"\[(\d+) hazards?\]", txt)
        ours = bool(re.search(rf"\b{re.escape(entry)}\b", txt)) or bool(files and any(f in files for f, _ in locs))
        reports.append({"severity": sev, "head": head[:220], "ours": ours,
                        "kind": report_kind(tool or "", head), "locs": locs[:8], "pcs": pcs[:8],
                        "hazards": sum(int(x) for x in hz) if hz else None})
    summ = re.search(r"ERROR SUMMARY: (\d+) error", text)
    rsumm = re.search(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", text)
    ours = [r for r in reports if r["ours"]]
    counted = [r for r in ours if r["kind"] is not None]
    sev_count = {s: sum(r["severity"] == s for r in counted) for s in ("error", "warning", "info")}
    return {"n_reports": len(reports), "n_ours": len(ours), "by_severity": sev_count,
            "not_ours": len(reports) - len(ours), "api_errors": api_errors[:5],
            "error_summary": int(summ.group(1)) if summ else None,
            "racecheck_summary": [int(x) for x in rsumm.groups()] if rsumm else None,
            "heads": [r["head"] for r in ours][:6],
            "locations": sorted({f"{f}:{ln}" for r in ours for f, ln in r["locs"]})[:10],
            "reports": [r for r in counted][:60],
            "caught": sev_count["error"] + sev_count["warning"] > 0,
            "info_only": sev_count["error"] + sev_count["warning"] == 0 and sev_count["info"] > 0}
