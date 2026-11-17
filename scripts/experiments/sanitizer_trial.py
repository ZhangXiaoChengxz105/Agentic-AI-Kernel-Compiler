"""Trial: can operator-agnostic GPU checkers catch the bugs our tests miss, without false alarms on correct kernels?

Tools:
- compute-sanitizer (CUDA toolkit):
  - memcheck, with --padding so that an overflow into a neighbouring buffer is still reported;
  - racecheck, at --racecheck-detect-level info, so that INFO hazards are recorded too;
  - synccheck;
  - initcheck, on global and shared memory.
- GPU-FPX detector (NVBit), if third_party/gpu-fpx-build/.../detector.so exists and passes a positive control.

Kernels (all CUDA; mutants are regenerated with akc.evaluation.mutants; nothing under akc/ is changed):
- correct    : the 7 correct bases. Any report here is a false alarm. They are softmax_opt, layernorm_opt, gemm_opt,
               glm_A_r1, glm_B1_r3, glm_B2_r1 and glm_B3_r1.
- racy_base  : glm_B1_r1, a GLM draft with a known shared-memory race.
- injected   : the 11 hand-injected CUDA bugs (workloads/*/manifest.json).
- race_oob   : the 14 mutants that only VOLTA caught: RACE / OUT_OF_BOUNDS rows of
               results/eval/mutants_volta_misses.json, without glm_B1_r1, whose base is racy.
- shift_only : the 8 mutants where the warp max shuffle starts at 8 instead of 16. They have 5 distinct sources,
               because const 16->8 and shfl_off 16->8 make the same edit on a line.

The launcher (`launch` sub-command) is a small driver-API launcher written here, not runners.gpu, because:
- every tensor is its own exact-size cuMemAlloc. The test harness surrounds outputs with canaries, which would hide
  out-of-bounds writes from memcheck;
- no PyTorch GPU kernels run, so every report comes from our kernel. Reports are still filtered by the entry name.
Inputs, the float64 golden (computed on the CPU) and the tolerances come from akc.task.reference.OPS. Outputs are
checked with akc.validation.runners.check.compare.

Usage:
  python scripts/experiments/sanitizer_trial.py list [--build]        # CPU: list kernels (and build their PTX)
  python scripts/experiments/sanitizer_trial.py run [--tools memcheck,racecheck,synccheck,initcheck,gpufpx]
         [--groups correct,...] [--jobs 6] [--out results/eval/sanitizer_trial.json]           # GPU node only
  python scripts/experiments/sanitizer_trial.py launch spec.json out.json    # the child process the tools wrap
  python scripts/experiments/sanitizer_trial.py reparse [results.json]       # CPU: re-parse saved logs in place
e.g.  sbatch --time=00:20:00 scripts/slurm/gpu_job.sbatch python scripts/experiments/sanitizer_trial.py run
Files: PTX and specs in build/sanitizer_trial/, raw tool logs in logs/sanitizer_trial/, summary in
results/eval/sanitizer_trial.json.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRIAL_BUILD = ROOT / "build" / "sanitizer_trial"
TRIAL_LOGS = ROOT / "logs" / "sanitizer_trial"
DEFAULT_OUT = ROOT / "results" / "eval" / "sanitizer_trial.json"
DEFAULT_CS = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/compute-sanitizer/compute-sanitizer"
NVBIT_TOOLS = ROOT / "third_party" / "gpu-fpx-build" / "nvbit_release" / "tools"
FPX_SO = NVBIT_TOOLS / "GPU-FPX" / "detector" / "detector.so"
INSTR_COUNT_SO = NVBIT_TOOLS / "instr_count" / "instr_count.so"

CORRECT = ["softmax_opt", "layernorm_opt", "gemm_opt", "glm_A_r1", "glm_B1_r3", "glm_B2_r1", "glm_B3_r1"]
RACY_BASE = "glm_B1_r1"
SHIFT_ONLY = [("glm_B1_r3", "const", 65), ("glm_B1_r3", "shfl_off", 65), ("glm_B2_r1", "const", 9),
              ("glm_B2_r1", "shfl_off", 9), ("glm_B3_r1", "const", 29), ("glm_B3_r1", "shfl_off", 29),
              ("glm_A_r1", "shfl_off", 65), ("glm_B1_r1", "shfl_off", 52)]
GROUPS = ("correct", "racy_base", "injected", "race_oob", "shift_only")

# Small shapes: the tools slow kernels down a lot, and every bug here shows at small sizes.
# N = 1000 is not a multiple of the 128-thread block; N = 8193 takes the GLM kernels' large-row path (N > 8192).
ROW_SHAPES = [{"rows": 64, "N": 256}, {"rows": 64, "N": 1000}, {"rows": 4, "N": 8193}]
GEMM_SHAPES = [{"M": 33, "N": 33, "K": 40}, {"M": 64, "N": 64, "K": 64}]
# 64 rows: the spike distribution puts the outlier of row r at warp lane r % 32, so 64 rows cover every lane twice.
VALUE_SHAPES = [{"rows": 64, "N": 256}, {"rows": 64, "N": 1000}]

SANITIZER_FLAGS = {
    "memcheck": ["--tool", "memcheck", "--padding", "256", "--destroy-on-device-error", "kernel"],
    "racecheck": ["--tool", "racecheck", "--racecheck-detect-level", "info"],
    "synccheck": ["--tool", "synccheck"],
    "initcheck": ["--tool", "initcheck", "--initcheck-address-space", "all"],
}
SANITIZERS = tuple(SANITIZER_FLAGS)
RUN_TIMEOUT_S = 300


@dataclass
class Kernel:
    label: str
    group: str
    workload: str
    src_file: str
    block: tuple
    members: list = field(default_factory=list)   # mutant ids this source stands for (shift_only dedupe)
    base: str | None = None
    desc: str = ""
    ptx: str = ""
    entry: str = ""
    build_error: str = ""


# ---------------------------------------------------------------------------------------------------------------
# kernel collection and PTX build (CPU)
# ---------------------------------------------------------------------------------------------------------------
def collect(groups=GROUPS) -> list[Kernel]:
    from akc.evaluation.mutants import BASES, HELDOUT, mutants
    from akc.task.workload import load_workload
    where = {lab: (wl, ROOT / rel) for lab, wl, rel in BASES + HELDOUT}
    block_of = lambda wl: tuple(int(b) for b in load_workload(wl).launch["block"])  # noqa: E731
    src_dir = TRIAL_BUILD / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    out: list[Kernel] = []

    def base_kernel(lab, group):
        wl, p = where[lab]
        return Kernel(lab, group, wl, str(p), block_of(wl), [lab], None, "unmodified base")

    def mutant_src(base, op, line):
        wl, p = where[base]
        ms = [m for m in mutants(p.read_text(), "cuda", 2) if m.op == op and m.line == line]
        if len(ms) != 1:
            raise RuntimeError(f"cannot regenerate mutant {base} {op} L{line} ({len(ms)} matches)")
        return wl, ms[0]

    if "correct" in groups:
        out += [base_kernel(lab, "correct") for lab in CORRECT]
    if "racy_base" in groups:
        out.append(base_kernel(RACY_BASE, "racy_base"))
    if "injected" in groups:
        for wl_name in ("softmax", "layernorm", "gemm"):
            wl = load_workload(wl_name)
            for v in wl.bugs("cuda"):
                out.append(Kernel(v.name, "injected", wl_name, str(v.file), v.block, [v.name], None,
                                  f"{v.bug.get('category')}: {v.bug.get('change')}"))
    if "race_oob" in groups:
        rows = json.loads((ROOT / "results" / "eval" / "mutants_volta_misses.json").read_text())["rows"]
        sel = [r for r in rows if r["base"] != RACY_BASE and set(r["statuses"]) & {"RACE", "OUT_OF_BOUNDS"}]
        if len(sel) != 14:
            raise RuntimeError(f"expected 14 RACE/OUT_OF_BOUNDS rows, found {len(sel)}")
        for r in sel:
            wl, m = mutant_src(r["base"], r["op"], r["line"])
            lab = f"{r['base']}_{m.op}_L{m.line}"
            f = src_dir / f"{lab}.cu"
            f.write_text(m.src)
            out.append(Kernel(lab, "race_oob", wl, str(f), block_of(wl), [lab], r["base"],
                              f"{m.desc} (VOLTA: {','.join(sorted(set(r['statuses'])))})"))
    if "shift_only" in groups:
        by_src: dict[str, Kernel] = {}
        for base, op, line in SHIFT_ONLY:
            wl, m = mutant_src(base, op, line)
            mid = f"{base}_{op}_L{line}"
            h = hashlib.sha1(m.src.encode()).hexdigest()[:10]
            if h in by_src:
                by_src[h].members.append(mid)
                continue
            lab = f"{base}_shift_L{line}"
            f = src_dir / f"{lab}.cu"
            f.write_text(m.src)
            by_src[h] = Kernel(lab, "shift_only", wl, str(f), block_of(wl), [mid], base, m.desc)
        out += list(by_src.values())
    return out


def build(kernels: list[Kernel], jobs: int = 8) -> None:
    from akc.backend.ptx import CompileError, compile_to_ptx

    def one(k: Kernel):
        try:
            b = compile_to_ptx(k.src_file, TRIAL_BUILD / "ptx" / k.label, mode="gpu")   # -lineinfo
            if len(b.kernels) != 1:
                raise RuntimeError(f"expected one entry, found {b.kernels}")
            k.ptx, k.entry = str(b.ptx), b.kernels[0]
        except (CompileError, RuntimeError, FileNotFoundError) as e:
            k.build_error = str(e)[-800:]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, kernels))


# ---------------------------------------------------------------------------------------------------------------
# child: minimal driver-API launcher (the process the tools wrap)
# ---------------------------------------------------------------------------------------------------------------
class _Dev:
    """Stands in for a tensor in OpSpec.kernel_args: only data_ptr() is used."""

    def __init__(self, ptr):
        self.ptr = int(ptr)

    def data_ptr(self):
        return self.ptr


def launch_main(spec_path: str, out_path: str) -> int:
    import torch
    from cuda.bindings import driver as cu

    from akc.task.reference import OPS
    from akc.task.workload import load_workload
    from akc.validation.runners.check import SENTINEL_BITS, compare

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            _, name = cu.cuGetErrorName(err)
            raise RuntimeError(f"{what}: {name.decode() if name else err}")
        return res[1] if isinstance(res, tuple) and len(res) > 1 else None

    spec = json.loads(Path(spec_path).read_text())
    op, wl = OPS[spec["workload"]], load_workload(spec["workload"])
    res = {"tests": [], "error": None}
    save = lambda: Path(out_path).write_text(json.dumps(res))  # noqa: E731
    try:
        ck(cu.cuInit(0), "cuInit")
        dev = ck(cu.cuDeviceGet(0), "cuDeviceGet")
        ctx = ck(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")   # primary: shared with cudart tools
        ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        mod = ck(cu.cuModuleLoadData(Path(spec["ptx"]).read_bytes() + b"\0"), "cuModuleLoadData")
        fn = ck(cu.cuModuleGetFunction(mod, spec["entry"].encode()), "cuModuleGetFunction")
    except Exception as e:  # noqa: BLE001
        res["error"] = f"setup: {e}"
        save()
        return 0
    dead = None
    for t in spec["tests"]:
        if dead:
            res["tests"].append({**t, "status": "skipped", "message": dead})
            continue
        allocs = []
        try:
            shape = t["shape"]
            inputs = {k: v.contiguous() for k, v in op.make_inputs(shape, t["dist"], t["seed"]).items()}
            d_in = {}
            for name, v in inputs.items():
                p = ck(cu.cuMemAlloc(v.numel() * 4), "cuMemAlloc")
                allocs.append(p)
                ck(cu.cuMemcpyHtoD(p, v.data_ptr(), v.numel() * 4), "cuMemcpyHtoD")
                d_in[name] = _Dev(p)
            golden = op.golden(inputs)
            n = golden.numel()
            bits = torch.full((n,), SENTINEL_BITS, dtype=torch.int32)         # unwritten outputs stay NaN-sentinel
            po = ck(cu.cuMemAlloc(n * 4), "cuMemAlloc")
            allocs.append(po)
            ck(cu.cuMemcpyHtoD(po, bits.data_ptr(), n * 4), "cuMemcpyHtoD")
            args = op.kernel_args(d_in, _Dev(po), shape)
            grid, blk = wl.launch_dims(shape, tuple(spec["block"]))
            objs = [ctype(value) for value, ctype in args]
            ptrs = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
            t0 = time.perf_counter()
            ck(cu.cuLaunchKernel(fn, *grid, *blk, 0, 0, ctypes.addressof(ptrs), 0), "cuLaunchKernel")
            ck(cu.cuCtxSynchronize(), "cuCtxSynchronize")
            ms = (time.perf_counter() - t0) * 1e3
            ck(cu.cuMemcpyDtoH(bits.data_ptr(), po, n * 4), "cuMemcpyDtoH")
            scale = op.magnitude(inputs) if op.magnitude else None
            r = compare(bits, bits.view(torch.float32), golden, atol=op.atol, rtol=op.rtol, scale=scale)
            res["tests"].append({**t, "status": "ok", "passed": r.passed, "reason": r.reason, "n_bad": r.n_bad,
                                 "n_nonfinite": r.n_nonfinite, "max_abs_err": r.max_abs_err, "launch_ms": ms})
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:300]
            res["tests"].append({**t, "status": "cuda_error", "message": msg})
            alive = cu.cuCtxSynchronize()[0] == cu.CUresult.CUDA_SUCCESS     # memcheck may stop only the kernel
            dead = None if alive else msg
        finally:
            for p in allocs:
                cu.cuMemFree(p)
        print(f"@@ {t['shape']} {t['dist']}: {res['tests'][-1].get('reason', res['tests'][-1]['status'])}", flush=True)
        save()
    cu.cuModuleUnload(mod)
    cu.cuDevicePrimaryCtxRelease(dev)      # context teardown: NVBit tools print their end-of-run report here
    save()
    return 0


# ---------------------------------------------------------------------------------------------------------------
# parent: run the tools, parse their reports
# ---------------------------------------------------------------------------------------------------------------
_SKIP_HEAD = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "LEAK SUMMARY", "Target application")


def parse_sanitizer(text: str, entry: str) -> dict:
    """Split the compute-sanitizer log into report blocks; keep those that name our kernel."""
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
        locs = sorted({f"{Path(f).name}:{ln}" for f, ln in re.findall(r" in (\S+?\.cu[h]?):(\d+)", txt)})
        reports.append({"severity": sev, "head": head[:220], "ours": bool(re.search(rf"\b{re.escape(entry)}\b", txt)),
                        "locations": locs[:6]})
    summ = re.search(r"ERROR SUMMARY: (\d+) error", text)
    rsumm = re.search(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", text)
    ours = [r for r in reports if r["ours"]]
    sev_count = {s: sum(r["severity"] == s for r in ours) for s in ("error", "warning", "info")}
    return {"n_reports": len(reports), "n_ours": len(ours), "by_severity": sev_count,
            "not_ours": len(reports) - len(ours), "api_errors": api_errors[:5],
            "error_summary": int(summ.group(1)) if summ else None,
            "racecheck_summary": [int(x) for x in rsumm.groups()] if rsumm else None,
            "heads": [r["head"] for r in ours][:6],
            "locations": sorted({loc for r in ours for loc in r["locations"]})[:10],
            "caught": sev_count["error"] + sev_count["warning"] > 0,
            "info_only": sev_count["error"] + sev_count["warning"] == 0 and sev_count["info"] > 0}


# The location is "<file>:<line>", or "/unknown_path in [<kernel>]:0" when NVBit has no line info for the
# instruction (seen for the GLM kernels, whose reductions are inlined __device__ helpers), so it may hold spaces.
_FPX_RE = re.compile(r"#GPU-FPX LOC-EXCEP INFO: (Warning: )?in kernel \[([^\]]*)\], (?:very small quantity \()?(\w+)\)? "
                     r"found @ (.+?) \[(FP\d+)\][ \t]*$", re.M)


def parse_fpx(text: str, entry: str) -> dict:
    found = []
    for m in _FPX_RE.finditer(text):
        loc = m.group(4)
        loc = "no line info" if loc.startswith("/unknown_path") else Path(loc).name
        found.append({"kernel": m.group(2), "exc": m.group(3), "loc": loc, "fp": m.group(5)})
    ours = [f for f in found if f["kernel"] == entry]
    kinds = sorted({f["exc"] for f in ours})
    return {"n_reports": len(found), "n_ours": len(ours), "kinds": kinds,
            "reports": sorted({f"{f['exc']}@{f['loc']}" for f in ours})[:10],
            "caught": any(k in ("NaN", "INF", "DIV0") for k in kinds),
            "instrumented": "Running #GPU-FPX: kernel [" in text}


def tests_for(k: Kernel, dists: list[str], value_shapes: bool = False) -> list[dict]:
    shapes = GEMM_SHAPES if k.workload == "gemm" else (VALUE_SHAPES if value_shapes else ROW_SHAPES)
    return [{"shape": dict(s), "dist": d, "seed": i} for d in dists for i, s in enumerate(shapes)]


def run_one(k: Kernel, tool: str, dists: list[str], value_shapes: bool, cs: str, tag: str = "") -> dict:
    name = f"{k.label}.{tool}{('.' + tag) if tag else ''}"
    spec, out, log = TRIAL_BUILD / "specs" / f"{name}.json", TRIAL_BUILD / "out" / f"{name}.json", TRIAL_LOGS / f"{name}.log"
    for p in (spec, out, log):
        p.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    spec.write_text(json.dumps({"workload": k.workload, "ptx": k.ptx, "entry": k.entry, "block": list(k.block),
                                "tests": tests_for(k, dists, value_shapes)}))
    child = [sys.executable, str(Path(__file__).resolve()), "launch", str(spec), str(out)]
    env = dict(os.environ)
    if tool in SANITIZERS:
        cmd = [cs, *SANITIZER_FLAGS[tool], "--show-backtrace", "no", "--print-limit", "30", "--log-file", str(log), *child]
    elif tool == "none":
        cmd = child
    else:                                   # gpufpx / nvbit_instr_count: LD_PRELOAD an NVBit tool
        cmd = child
        env["LD_PRELOAD"] = str(FPX_SO if tool == "gpufpx" else INSTR_COUNT_SO)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), ""
    secs = time.perf_counter() - t0
    if tool not in SANITIZERS:
        log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{stdout}\n--- stderr\n{stderr}")
    child_res = json.loads(out.read_text()) if out.exists() else {"error": "no output (child crashed)", "tests": []}
    row = {"label": k.label, "group": k.group, "tool": tool, "dists": dists, "seconds": round(secs, 1),
           "exit": rc, "child_error": child_res.get("error"), "log": str(log.relative_to(ROOT)),
           "tests": [{"shape": t["shape"], "dist": t["dist"], "status": t["status"],
                      "passed": t.get("passed"), "reason": t.get("reason", t.get("message"))} for t in child_res["tests"]]}
    if tool in SANITIZERS:
        row["result"] = parse_sanitizer(log.read_text() if log.exists() else "", k.entry)
    elif tool == "gpufpx":
        row["result"] = parse_fpx(stdout, k.entry)
    elif tool == "nvbit_instr_count":
        row["result"] = {"ok": rc == 0 and "instructions" in stdout.lower(), "tail": stdout.strip().splitlines()[-3:]}
    if tool != "none" and rc not in (0,):
        row["stderr_tail"] = (stderr or stdout).strip().splitlines()[-4:]
    return row


def summarize(kernels: list[Kernel], runs: list[dict]) -> dict:
    by_label = {k.label: k for k in kernels}
    s: dict = {}
    for r in runs:
        if r["tool"] not in SANITIZERS:
            continue
        k = by_label[r["label"]]
        g = s.setdefault(k.group, {}).setdefault(r["tool"], {"caught": 0, "info_only": 0, "total": 0, "caught_ids": []})
        n = len(k.members)
        g["total"] += n
        if r.get("result", {}).get("caught"):
            g["caught"] += n
            g["caught_ids"] += k.members
        elif r.get("result", {}).get("info_only"):
            g["info_only"] += n
    for grp in s.values():                          # any of the sanitizer tools
        ids = set().union(*[set(v["caught_ids"]) for v in grp.values()])
        grp["any_sanitizer"] = {"caught": len(ids), "total": max(v["total"] for v in grp.values()), "caught_ids": sorted(ids)}
    fpx = [r for r in runs if r["tool"] == "gpufpx"]
    if fpx:
        fs: dict = {}
        for r in fpx:
            k = by_label[r["label"]]
            d = "+".join(r["dists"])
            g = fs.setdefault(f"{k.group}/{d}", {"caught": 0, "total": 0, "caught_ids": []})
            g["total"] += len(k.members)
            if r.get("result", {}).get("caught"):
                g["caught"] += len(k.members)
                g["caught_ids"] += k.members
        s["gpufpx"] = fs
    return s


def run_main(a) -> int:
    t_start = time.perf_counter()
    tools = [t for t in a.tools.split(",") if t]
    groups = [g for g in a.groups.split(",") if g]
    cs = a.compute_sanitizer
    kernels = collect(groups)
    if a.labels:
        keep = set(a.labels.split(","))
        kernels = [k for k in kernels if k.label in keep]
    build(kernels)
    bad = [k for k in kernels if k.build_error]
    for k in bad:
        print(f"BUILD FAILED {k.label}: {k.build_error[:300]}")
    kernels = [k for k in kernels if not k.build_error]
    print(f"{len(kernels)} kernels built; tools {tools}; {a.jobs} parallel runs", flush=True)
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename, "compute_sanitizer": cs, "tools": tools, "groups": groups,
            "row_shapes": ROW_SHAPES, "gemm_shapes": GEMM_SHAPES, "value_shapes": VALUE_SHAPES,
            "sanitizer_flags": SANITIZER_FLAGS, "jobs": a.jobs}
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
        meta["sanitizer_version"] = subprocess.run([cs, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except OSError as e:
        meta["gpu"] = f"unknown ({e})"
    runs: list[dict] = []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(done=False):
        res = {"meta": {**meta, "wall_s": round(time.perf_counter() - t_start, 1), "complete": done},
               "kernels": [{kk: v for kk, v in asdict(k).items() if kk not in ("src_file",)} | {"src_file": str(Path(k.src_file).relative_to(ROOT))}
                           for k in kernels] + [{"label": k.label, "build_error": k.build_error} for k in bad],
               "runs": runs, "summary": summarize(kernels, runs)}
        out.write_text(json.dumps(res, indent=1))

    # 0) self-test without any tool: if this fails, the launcher is broken and nothing else is meaningful
    ref = next((k for k in kernels if k.label == "softmax_opt"), kernels[0])
    st = run_one(ref, "none", ["normal"], False, cs)
    meta["self_test"] = st
    print(f"self-test {ref.label}: exit {st['exit']}, tests {[t['reason'] for t in st['tests']]}", flush=True)
    if st["child_error"] or not st["tests"] or not all(t.get("passed") for t in st["tests"]):
        print("self-test failed; stopping")
        save(True)
        return 1

    # 1) compute-sanitizer tools
    jobs = []
    for tool in [t for t in tools if t in SANITIZERS]:
        for k in kernels:
            dists = ["normal", "outlier@128"] if k.group == "shift_only" else ["normal"]
            jobs.append((k, tool, dists, False))
    t_tool: dict[str, float] = {}

    def do(j):
        k, tool, dists, vs = j
        r = run_one(k, tool, dists, vs, cs)
        res = r.get("result", {})
        print(f"{tool:9s} {k.group:10s} {k.label:34s} {r['seconds']:6.1f}s exit={r['exit']} "
              f"ours={res.get('n_ours')} sev={res.get('by_severity')} {'CAUGHT' if res.get('caught') else ''} "
              f"{res.get('heads', [''])[:1]}", flush=True)
        return r
    with ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(do, jobs):
            runs.append(r)
            t_tool[r["tool"]] = t_tool.get(r["tool"], 0.0) + r["seconds"]
            if len(runs) % 20 == 0:
                save()
    meta["tool_cpu_seconds"] = {t: round(v, 1) for t, v in t_tool.items()}
    save()

    # 2) GPU-FPX: NVBit sanity, positive control, then the shift-only mutants and their bases
    if "gpufpx" in tools:
        t0 = time.perf_counter()
        fpx_meta = {"detector_so": str(FPX_SO.relative_to(ROOT)), "exists": FPX_SO.exists()}
        by = {k.label: k for k in kernels}
        if not FPX_SO.exists():
            fpx_meta["status"] = "detector.so not built"
        else:
            if INSTR_COUNT_SO.exists() and ref:
                r = run_one(ref, "nvbit_instr_count", ["normal"], True, cs)
                fpx_meta["nvbit_instr_count"] = {"exit": r["exit"], **r["result"], "stderr_tail": r.get("stderr_tail")}
                print(f"NVBit instr_count sanity: {fpx_meta['nvbit_instr_count']}", flush=True)
            ctrl = by.get("softmax_bug_nomax")
            ok = False
            if ctrl:
                r = run_one(ctrl, "gpufpx", ["large"], True, cs, tag="large")
                runs.append(r)
                fpx_meta["control"] = {"label": ctrl.label, "dist": "large", "exit": r["exit"], **r["result"],
                                       "stderr_tail": r.get("stderr_tail"), "tests": r["tests"]}
                ok = r["result"]["instrumented"] and r["result"]["caught"]
                print(f"GPU-FPX control: {fpx_meta['control']}", flush=True)
            fpx_meta["status"] = "working" if ok else "control failed: GPU-FPX did not instrument or report"
            if ok:
                fjobs = []
                for k in kernels:
                    if k.group == "shift_only":
                        for d in ("normal", "outlier@128"):
                            fjobs.append((k, d))
                            if k.base in by and (by[k.base], d) not in fjobs:
                                fjobs.append((by[k.base], d))
                for lab, d in (("softmax_opt", "large"), ("layernorm_bug_onepass", "offset"), ("layernorm_opt", "offset")):
                    if lab in by:
                        fjobs.append((by[lab], d))

                def fdo(j):
                    k, d = j
                    r = run_one(k, "gpufpx", [d], True, cs, tag=d)
                    print(f"gpufpx {k.group:10s} {k.label:34s} {d:7s} {r['seconds']:6.1f}s exit={r['exit']} "
                          f"{r['result']['kinds']} {r['result']['reports'][:3]} tests={[t['reason'] for t in r['tests']]}", flush=True)
                    return r
                with ThreadPoolExecutor(a.jobs) as ex:
                    runs.extend(ex.map(fdo, fjobs))
        fpx_meta["seconds"] = round(time.perf_counter() - t0, 1)
        meta["gpufpx"] = fpx_meta
    save(True)
    s = summarize(kernels, runs)
    print(json.dumps({g: {t: {kk: v for kk, v in d.items() if kk != "caught_ids"} for t, d in v.items()} for g, v in s.items()}, indent=1))
    print(f"saved {out} ({time.perf_counter() - t_start:.0f}s)")
    return 0


def reparse_main(path: str) -> int:
    """Re-parse the saved tool logs of a finished run (CPU only) and rewrite results + summary in place."""
    p = Path(path)
    d = json.loads(p.read_text())
    entry = {k["label"]: k.get("entry", "") for k in d["kernels"]}
    for r in d["runs"]:
        log = ROOT / r["log"]
        if not log.exists():
            continue
        text = log.read_text()
        if r["tool"] in SANITIZERS:
            r["result"] = parse_sanitizer(text, entry[r["label"]])
        elif r["tool"] == "gpufpx":
            r["result"] = parse_fpx(text.split("--- stderr")[0], entry[r["label"]])
    ctrl = d["meta"].get("gpufpx", {}).get("control")
    if ctrl:
        log = ROOT / next(r["log"] for r in d["runs"] if r["tool"] == "gpufpx" and r["label"] == ctrl["label"])
        ctrl.update(parse_fpx(log.read_text().split("--- stderr")[0], entry[ctrl["label"]]))
    kernels = [Kernel(**{f: k[f] for f in ("label", "group", "workload", "src_file", "block", "members", "base",
                                           "desc", "ptx", "entry")}) for k in d["kernels"] if "group" in k]
    d["summary"] = summarize(kernels, d["runs"])
    d["meta"]["reparsed"] = datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(d, indent=1))
    print(f"re-parsed {len(d['runs'])} runs; saved {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("--build", action="store_true")
    pl.add_argument("--groups", default=",".join(GROUPS))
    pr = sub.add_parser("run")
    pr.add_argument("--tools", default="memcheck,racecheck,synccheck,initcheck,gpufpx")
    pr.add_argument("--groups", default=",".join(GROUPS))
    pr.add_argument("--labels", default="")
    pr.add_argument("--jobs", type=int, default=6)
    pr.add_argument("--out", default=str(DEFAULT_OUT))
    pr.add_argument("--compute-sanitizer", default=shutil.which("compute-sanitizer") or DEFAULT_CS)
    pc = sub.add_parser("launch")
    pc.add_argument("spec")
    pc.add_argument("out")
    pp = sub.add_parser("reparse")
    pp.add_argument("json", nargs="?", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    if a.cmd == "launch":
        return launch_main(a.spec, a.out)
    if a.cmd == "reparse":
        return reparse_main(a.json)
    if a.cmd == "run":
        return run_main(a)
    ks = collect([g for g in a.groups.split(",") if g])
    if a.build:
        build(ks)
    for k in ks:
        print(f"{k.group:10s} {k.label:34s} {k.workload:9s} block={k.block} members={k.members} "
              f"{k.entry or ''} {('BUILD ERROR ' + k.build_error[:200]) if k.build_error else ''} {k.desc[:60]}")
    print(f"{len(ks)} kernels, {sum(len(k.members) for k in ks)} kernel ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
