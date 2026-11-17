"""Static compiler checks of a tirx-lite candidate (the "CAKE compiler checks" of our CAKE substitute), plus the
CPU-only dynamic stage (TIRx NumSim racecheck / synccheck), run in ONE child process per candidate.

    gate_shapes(workload, src) -> [shape]          the gate's small shapes (inside the manifest domain)
    numsim_shape(workload, src) -> shape           the one small shape NumSim checks on a CPU-only node
    run_static(workload, module_path, shapes, out_dir, numsim=None, timeout_s=None) -> dict

Why a child process: importing tvm takes ~5 s and the module is agent-written Python that runs while it is traced
(it could hang or crash). Same pattern as changeaware.ir_worker: the parent safety-scans first
(backend.tirx.safety_scan), the child loads the module with backend.tirx.load_module and traces the manifest
factory; a SIGALRM bounds every shape.

Per shape, in this order (`stages` records seconds and status of each):
  trace          make_kernel(rows, n) / (m, n, k). txl.kernel(check_ir=True) is the default, so the low-level IR
                 contract (no direct global / shared TensorLoad / TensorStore, no tile ops, no unlisted
                 func_call, ...) is checked while tracing and raises LowLevelIRContractError. Other exceptions are
                 trace errors; their line in the module comes from the traceback.
  ir_contract    tirx_kernels.tirx_lite.low_level_ir.inspect_low_level_ir(func): the same contract again, explicitly,
                 so a module that turns it off with @txl.kernel(check_ir=False) is still checked.
                 Violations are located at Python lines by tracing again with the check switched to record-only and
                 matching the violating statements (changeaware.irdiff.flatten gives each statement's line).
  well_formed    tvm_tirx.tirx.analysis.check_well_formed(func, assert_mode=False)
  verify_memory  tvm_tirx.tirx.analysis.verify_memory(func)  (no illegal host-side memory access)
  compile        backend.tirx.compile_sm80 (TIRx pipeline + NVRTC for sm_80). Its generated CUDA
                 (`Kernel.source()` is the same text) is written to <out_dir>/<label>_<shape>.cu, with the entry name,
                 the tensor parameter order (volta.tirx.kernel_tensor_params), grid, block and dynamic shared bytes
                 (volta.tirx.launch_info): what the compute-sanitizer stage needs to launch it.
Measured on the login node: tvm import ~5.4 s; per shape trace 0.05-0.2 s, the four IR checks ~3 ms, compile
1.5-2.2 s.

NumSim stage (only when `numsim` is given and every static check passed): tirx_harness.numsim.checkers.racecheck and
synccheck of the kernel traced at numsim["shape"], on the arrays the parent wrote as .npy files (inputs and
zero-filled outputs). Each finding keeps its status (error / review / incomplete), kind, message and the module
lines of its source spans. Needs RUSTUP_TOOLCHAIN=1.95.0 (set by the parent). Cost: ~5 s per checker on a small
softmax shape once NumSim's native engine is built (the first synccheck on a machine built it in ~57 s).

Result dict (JSON): import_s, wall_s, shapes [per shape: shape, label, stages, cuda, entry, params, grid, block,
smem, launch_error], errors [stage, shape, message, lines], numsim {shape, racecheck, synccheck} | None,
error (the child itself failed: timeout, crash, no tvm) | None.
Child: python -m akc.validation.compiler_checks.tirx_static job.json out.json
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from ...paths import BUILD, ROOT

SHAPE_TIMEOUT_S = 60          # SIGALRM per shape (trace + checks + compile)
NUMSIM_TIMEOUT_S = 900        # racecheck + synccheck (a first native-engine build takes minutes)
STAGES = ("trace", "ir_contract", "well_formed", "verify_memory", "compile")
NUMSIM_CHECKS = ("racecheck", "synccheck")


# ---- shapes (parent side, no tvm) --------------------------------------------------------------------------------
def _tile(workload: str, src: str) -> int:
    from ...task.workload import load_workload
    from ..changeaware.generate import extract_tiles
    wl = load_workload(workload)
    return (extract_tiles(src or "", "tirx") or list(wl.tiles))[0]


def gate_shapes(workload: str, src: str) -> list[dict]:
    """The compiler-checks gate's shapes, small so the sanitizer tools stay cheap, all inside the manifest domain:
      row ops (rows, N): rows = 4 with N in {T + 1, 1000} (T = the candidate's tile, else the manifest's), plus
                         rows = 2, N = v + 1 for each size-regime value v of the candidate (extract_thresholds: the
                         kernel switches code path between v and v + 1; v + 1 is the large-size side)
      gemm (M, N, K)   : 33 x 33 x 40 (not multiples of the 16 tile: partial tiles in every dimension) and
                         64 x 64 x 64 (whole tiles, 4 x 4 CTAs, 4 k-steps)
    """
    from ...task.workload import load_workload
    from ..changeaware.generate import extract_thresholds
    wl = load_workload(workload)
    keys = wl.shape_keys
    if wl.launch["kind"] == "row":
        t = _tile(workload, src)
        shapes = [{"rows": 4, "N": t + 1}, {"rows": 4, "N": 1000}]
        for key, v in extract_thresholds(src or "", "tirx", keys, wl.launch["kind"], wl.domain):
            shapes.append({"rows": 2, "N": 1000, key: v + 1})
    else:
        shapes = [{"M": 33, "N": 33, "K": 40}, {"M": 64, "N": 64, "K": 64}]
    out = []
    for s in shapes:
        s = wl.clip({k: max(1, int(v)) for k, v in s.items() if k in keys})
        if s not in out and wl.in_domain(s):
            out.append(s)
    return out


def numsim_shape(workload: str, src: str) -> dict:
    """The one small shape of the CPU-only NumSim stage (NumSim simulates every thread, so it must be tiny):
    row ops rows = 2, N = T + 1 (two CTAs, one partial tile); gemm 12 x 12 x 20 (one CTA whose 16 x 16 tile
    overhangs both edges, two k-steps so a missing barrier between steps can race)."""
    from ...task.workload import load_workload
    wl = load_workload(workload)
    if wl.launch["kind"] == "row":
        s = {"rows": 2, "N": min(_tile(workload, src) + 1, 2048)}
    else:
        s = {"M": 12, "N": 12, "K": 20}
    return wl.clip(s)


def shape_label(shape: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in shape.items())


# ---- parent side ---------------------------------------------------------------------------------------------------
def run_static(workload: str, module_path: str | Path, shapes: list[dict], out_dir: str | Path, label: str = "cand",
               numsim: dict | None = None, timeout_s: float | None = None) -> dict:
    """Run the static checks (and, with `numsim` = {"shape", "arrays": {tensor: .npy path}}, the NumSim stage) in a
    child process. Returns the child's result dict; on a timeout / crash, `error` says why and the shapes finished
    so far are kept."""
    from ...backend.tirx import safety_scan
    t0 = time.perf_counter()
    problems = safety_scan(Path(module_path).read_text())
    if problems:
        return {"import_s": 0.0, "wall_s": time.perf_counter() - t0, "shapes": [], "numsim": None, "error": None,
                "errors": [{"stage": "safety_scan", "shape": None, "lines": [],
                            "message": "refused to import the module: " + "; ".join(problems)[:600]}]}
    timeout_s = timeout_s or (60 + (SHAPE_TIMEOUT_S // 2) * len(shapes) + (NUMSIM_TIMEOUT_S if numsim else 0))
    (BUILD / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD / "tmp", prefix="gate-") as tmp:
        job, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job.write_text(json.dumps({"workload": workload, "module": str(module_path), "label": label, "shapes": shapes,
                                   "out_dir": str(out_dir), "numsim": numsim}))
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if numsim:
            env.update(RUSTUP_TOOLCHAIN="1.95.0", CARGO_BUILD_JOBS="4")
        err = None
        try:
            p = subprocess.run([sys.executable, "-m", "akc.validation.compiler_checks.tirx_static", str(job), str(out)],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
            if p.returncode != 0:
                err = f"static-check worker exited {p.returncode}: {(p.stderr or p.stdout)[-800:]}"
        except subprocess.TimeoutExpired:
            err = f"static-check worker timed out after {timeout_s:.0f}s"
        res = json.loads(out.read_text()) if out.exists() else {"shapes": [], "errors": [], "numsim": None}
    res.setdefault("errors", [])
    res["error"] = res.get("error") or err
    res["wall_s"] = time.perf_counter() - t0
    return res


# ---- child side ----------------------------------------------------------------------------------------------------
class _Timeout(BaseException):          # not caught by the `except Exception` around agent code
    pass


def _on_alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def _tb_lines(exc: BaseException, module: str) -> list[int]:
    """Lines of the module file in the exception's traceback (innermost last)."""
    try:
        return [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == module and f.lineno]
    except Exception:  # noqa: BLE001
        return []


def _violation_lines(func, module: str, report) -> list[int]:
    """Python lines of the statements that hold the contract violations of `report` (best effort)."""
    import re
    lines: list[int] = []
    for v in report.violations:                  # a span "<file>:<line>:<col>[-<line>:<col>]" when TIRx has one
        m = re.match(r"^(.*):(\d+):(\d+)(?:-\d+:\d+)?$", v.span or "")
        if m and m.group(1) == module and int(m.group(2)) not in lines:
            lines.append(int(m.group(2)))
    try:
        from tirx_kernels.tirx_lite.low_level_ir import _LowLevelIRInspector

        from ..changeaware import irdiff
        want = {(v.kind, v.callee) for v in report.violations}
        for r in irdiff.flatten(func, module).recs:
            if r.line is None or r.kind in ("Function", "Else", "Region"):
                continue
            ins = _LowLevelIRInspector("root", frozenset())
            try:                                   # a header: only its own expressions, not its body
                for part in irdiff._header_parts(r.node) if r.kind in ("For", "If", "While") else [r.node]:
                    if part is not None:
                        ins(part)
            except Exception:  # noqa: BLE001
                continue
            if {(v.kind, v.callee) for v in ins.violations} & want and r.line not in lines:
                lines.append(r.line)
    except Exception:  # noqa: BLE001
        pass
    return sorted(lines)


def _trace_without_check(mod, meta, shape):
    """Trace again with the contract check switched to record-only (to get a Function that can be located)."""
    import tirx_kernels.tirx_lite.low_level_ir as lli

    from ...backend.tirx import make_kernel
    orig = lli.check_low_level_ir
    lli.check_low_level_ir = lambda value, **kw: lli.inspect_low_level_ir(value, **kw)
    try:
        return make_kernel(mod, meta, shape)
    finally:
        lli.check_low_level_ir = orig


def _one_shape(wl, mod, module: str, label: str, shape: dict, out_dir: Path) -> tuple[dict, list[dict]]:
    from tirx_kernels.tirx_lite.low_level_ir import LowLevelIRContractError, inspect_low_level_ir
    from tvm_tirx.tirx.analysis import check_well_formed, verify_memory

    from ...backend.tirx import compile_sm80, make_kernel
    from ..volta.tirx import TirxBuildError, kernel_tensor_params, launch_info
    rec = {"shape": shape, "label": shape_label(shape), "stages": {}}
    errors: list[dict] = []

    def stage(name, t0, status="ok"):
        rec["stages"][name] = {"seconds": round(time.perf_counter() - t0, 4), "status": status}

    def fail(name, t0, message, lines=()):
        stage(name, t0, "error")
        errors.append({"stage": name, "shape": shape, "message": message[:2000], "lines": list(lines)})

    t0 = time.perf_counter()
    try:
        k = make_kernel(mod, wl.meta, shape)
        stage("trace", t0)
    except LowLevelIRContractError as e:
        stage("trace", t0, "error")
        t1 = time.perf_counter()
        lines = []
        try:
            lines = _violation_lines(_trace_without_check(mod, wl.meta, shape).func, module, e.report)
        except Exception:  # noqa: BLE001
            pass
        fail("ir_contract", t1, f"LowLevelIRContractError: {e}", lines)
        return rec, errors
    except _Timeout:
        raise
    except Exception as e:  # noqa: BLE001  (the agent's Python failed while tracing)
        fail("trace", t0, f"{type(e).__name__}: {str(e).strip()}", _tb_lines(e, module)[-1:])
        return rec, errors
    t0 = time.perf_counter()
    try:
        rep = inspect_low_level_ir(k.func)
        if rep.ok:
            stage("ir_contract", t0)
        else:
            fail("ir_contract", t0, rep.summary(), _violation_lines(k.func, module, rep))
    except Exception as e:  # noqa: BLE001
        fail("ir_contract", t0, f"{type(e).__name__}: {e}")
    t0 = time.perf_counter()
    try:
        if check_well_formed(k.func, assert_mode=False):
            stage("well_formed", t0)
        else:
            fail("well_formed", t0, "tvm_tirx check_well_formed: the traced function is not well formed")
    except Exception as e:  # noqa: BLE001
        fail("well_formed", t0, f"check_well_formed: {type(e).__name__}: {e}")
    t0 = time.perf_counter()
    try:
        if verify_memory(k.func):
            stage("verify_memory", t0)
        else:
            fail("verify_memory", t0, "tvm_tirx verify_memory: illegal host-side direct memory access")
    except Exception as e:  # noqa: BLE001
        fail("verify_memory", t0, f"verify_memory: {type(e).__name__}: {e}")
    if errors:
        return rec, errors
    t0 = time.perf_counter()
    try:
        ex = compile_sm80(k)
        src = ex.mod.imports[0].inspect_source("cuda")
        stage("compile", t0)
    except _Timeout:
        raise
    except Exception as e:  # noqa: BLE001
        fail("compile", t0, f"{type(e).__name__}: {str(e).strip()}", _tb_lines(e, module)[-1:])
        return rec, errors
    out_dir.mkdir(parents=True, exist_ok=True)
    cu = out_dir / f"{label}_{rec['label']}.cu"
    cu.write_text(src)
    rec["cuda"] = str(cu)
    try:
        grid, block, smem = launch_info(k)
        rec.update(grid=list(grid), block=list(block), smem=smem)
        rec["entry"], rec["params"] = kernel_tensor_params(k, src, wl.meta["tirx"]["tensors"])
    except TirxBuildError as e:
        rec["launch_error"] = e.message                # e.g. scalar parameters: the sanitizer cannot launch it
    except Exception as e:  # noqa: BLE001
        rec["launch_error"] = f"{type(e).__name__}: {e}"
    return rec, errors


def _op_lines(op: dict | None, module: str) -> tuple[int | None, int | None]:
    """(line, call-site line) of one NumSim operation: its source_span lists the frames outermost first, so the last
    span in the module is the statement itself and the first one the call in the kernel body (if different)."""
    spans = ((((op or {}).get("operation") or {}).get("source") or {}).get("source_span") or {})
    seq = spans.get("spans") if spans.get("kind") == "sequential" else [spans]
    lines = [s.get("line") for s in seq or [] if isinstance(s, dict) and s.get("source_name") == module
             and isinstance(s.get("line"), int)]
    if not lines:
        return None, None
    return lines[-1], (lines[0] if lines[0] != lines[-1] else None)


def _span_lines(obj, module: str) -> list[int]:
    """Lines of `module` in every source_span below obj (NumSim findings), in the order they appear."""
    out: list[int] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("kind") == "span" and o.get("source_name") == module and isinstance(o.get("line"), int):
                if o["line"] not in out:
                    out.append(o["line"])
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return out


def _numsim_finding(d: dict, module: str) -> dict:
    """One NumSim finding -> {status, kind, message, summary, lines, callers} (lines: the statements involved)."""
    det = d.get("details") or {}
    lines, callers, parts = [], [], []
    for side in ("prior", "current"):
        op = det.get(side)
        if not isinstance(op, dict):
            continue
        ln, via = _op_lines(op, module)
        if ln is not None and ln not in lines:
            lines.append(ln)
        if via is not None and via not in callers:
            callers.append(via)
        o = op.get("operation") or {}
        parts.append(f"{op.get('access_kind', '?')} by warp {o.get('global_warp_id', '?')} lane {op.get('lane', '?')}"
                     + (f" at line {ln}" if ln else "") + (f" (helper called from line {via})" if via else ""))
    if not lines:
        lines = _span_lines(det, module)
    space = (det.get("prior") or {}).get("space") if isinstance(det.get("prior"), dict) else None
    summary = (f"{det.get('access_pair', '').replace('_', '/')} conflict on {space or '?'} memory: "
               + " vs ".join(parts) + (f"; {det.get('ordering_failure', '').replace('_', ' ')}" if det.get(
                   "ordering_failure") else "")) if parts else str(d.get("message", ""))[:300]
    return {"status": d["status"], "kind": d["kind"], "message": str(d.get("message", ""))[:600],
            "summary": summary[:400], "lines": lines[:6], "callers": callers[:4]}


def _numsim(wl, mod, module: str, job: dict) -> dict:
    import numpy as np
    from tirx_harness.numsim.checkers import racecheck, synccheck

    from ...backend.tirx import make_kernel
    shape = job["shape"]
    res = {"shape": shape}
    try:
        k = make_kernel(mod, wl.meta, shape)
        arrays = {n: np.load(p) for n, p in job["arrays"].items()}
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"{type(e).__name__}: {str(e)[:600]}"}
    for name, fn in (("racecheck", racecheck), ("synccheck", synccheck)):
        t0 = time.perf_counter()
        try:
            r = fn(k.func, inputs={n: a.copy() for n, a in arrays.items()})
            finds = [_numsim_finding(f.to_dict(), module) for f in r.findings[:20]]
            res[name] = {"verdict": r.verdict, "seconds": round(time.perf_counter() - t0, 2), "findings": finds}
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e).strip()[:600]}"
            res[name] = {"verdict": "crash", "seconds": round(time.perf_counter() - t0, 2), "error": msg,
                         "lines": _tb_lines(e, module)[-1:]}
    return res


def _child(job_path: str, out_path: str) -> None:
    j = json.loads(Path(job_path).read_text())
    out = {"shapes": [], "errors": [], "numsim": None, "error": None, "import_s": None}
    save = lambda: Path(out_path).write_text(json.dumps(out, default=str))  # noqa: E731
    t0 = time.perf_counter()
    try:
        import tirx_kernels.tirx_lite  # noqa: F401
        import tvm_tirx.tirx.analysis  # noqa: F401

        from ...backend.tirx import load_module
        from ...task.workload import load_workload
        wl = load_workload(j["workload"])
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cannot run the tirx-lite checks here: {type(e).__name__}: {str(e)[:400]}"
        save()
        return
    out["import_s"] = round(time.perf_counter() - t0, 3)
    module = str(Path(j["module"]).resolve())
    t0 = time.perf_counter()
    try:
        mod = load_module(module, "akc_gate_candidate")
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"stage": "import", "shape": None, "message": f"{type(e).__name__}: {str(e).strip()[:1500]}",
                              "lines": _tb_lines(e, module)[-1:]})
        out["load_s"] = round(time.perf_counter() - t0, 3)
        save()
        return
    out["load_s"] = round(time.perf_counter() - t0, 3)
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
    for shape in j["shapes"]:
        if has_alarm:
            signal.alarm(SHAPE_TIMEOUT_S)
        try:
            rec, errs = _one_shape(wl, mod, module, j["label"], shape, Path(j["out_dir"]))
        except _Timeout:
            rec, errs = {"shape": shape, "label": shape_label(shape), "stages": {}}, [
                {"stage": "timeout", "shape": shape, "lines": [],
                 "message": f"tracing / compiling took longer than {SHAPE_TIMEOUT_S}s at this shape"}]
        finally:
            if has_alarm:
                signal.alarm(0)
        out["shapes"].append(rec)
        out["errors"] += errs
        save()
    if j.get("numsim") and not out["errors"]:
        if has_alarm:
            signal.alarm(NUMSIM_TIMEOUT_S)
        t0 = time.perf_counter()
        try:
            out["numsim"] = _numsim(wl, mod, module, j["numsim"])
        except _Timeout:
            out["numsim"] = {"shape": j["numsim"]["shape"], "error": f"NumSim took longer than {NUMSIM_TIMEOUT_S}s"}
        finally:
            if has_alarm:
                signal.alarm(0)
        out["numsim"]["seconds"] = round(time.perf_counter() - t0, 2)
    save()


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
