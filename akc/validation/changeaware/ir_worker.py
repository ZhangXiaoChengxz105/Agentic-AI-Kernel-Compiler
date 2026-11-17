"""Task 2.1 helper: run the IR change analysis (irdiff + classify_ir) of tirx-lite revisions in a child process.

Why a child process: importing tvm takes ~5 s and must not slow down CUDA callers, and a tirx-lite module is
agent-written Python that runs while it is traced (it could hang or crash). Each module is checked with
backend.tirx.safety_scan (the scan the NumSim and GPU runners use) before anything executes, then loaded with
backend.tirx.load_module and traced with the manifest's factory (make_kernel(rows, n)).

trace_shapes(workload, old_src, new_src) -> [shape]        where both versions are traced (<= MAX_SHAPES)
analyze(workload, old_src, new_src, shapes, timeout_s=60) -> result dict   (cached in this process)
prefetch(workload, items, timeout_s=60)                   one child for many (old, new, shapes); fills the cache
                                                          (the mutation benchmark: tvm is imported once per base)

Result dict (JSON):
  ok        False -> the IR path cannot be used; `error` says why (refused by the safety scan, the new version
            does not trace at some shape, timeout, no tvm_tirx, ...) and the caller falls back to the AST path
  hunks     merged IR hunks (irdiff.merge), labels (classify_ir, as dicts)
  shapes    shapes traced on both sides; skipped {shape: reason} where the OLD version (or neither) does not trace
  per_shape {shape: raw hunk count before merging}, printer ("script" | "fallback"), seconds
Child: python -m akc.validation.changeaware.ir_worker job.json out.json   (out.json is rewritten after every job,
so a timeout keeps the jobs already finished; each job also has its own SIGALRM limit).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from ...paths import BUILD, ROOT

TIMEOUT_S = 60
MAX_SHAPES = 8
_CACHE: dict[str, dict] = {}


# ---- shapes ---------------------------------------------------------------------------------------------------
def trace_shapes(workload: str, old_src: str | None, new_src: str, max_shapes: int = MAX_SHAPES) -> list[dict]:
    """Shapes to trace, all inside the declared domain, rows = 2 (the IR does not depend on rows beyond the grid):
    the benchmark shape (perf.shape), N in {1, T-1, T+1} for the first tile T, both sides v, v+1 of every size regime
    switch (extract_thresholds of either version), N = 2T+1, then the same sizes for further tiles; deduplicated,
    at most max_shapes (regimes come before 2T+1 so a second code path is not cut)."""
    from ...task.workload import load_workload
    from .generate import extract_thresholds, extract_tiles
    wl = load_workload(workload)
    keys = wl.shape_keys
    tiles = sorted(set(extract_tiles(new_src, "tirx")) | set(extract_tiles(old_src or "", "tirx"))) or list(wl.tiles)
    base = dict(wl.perf_shape or wl.meta["fixed_suite"][0])
    if "rows" in base:
        base["rows"] = 2
    size = "N" if "N" in keys else keys[-1]
    t = tiles[0]
    shapes = [base] + [{**base, size: n} for n in (1, t - 1, t + 1)]
    regimes = []
    for src in (new_src, old_src or ""):
        for r in extract_thresholds(src, "tirx", keys, wl.launch["kind"], wl.domain):
            if r not in regimes:
                regimes.append(r)
    for key, v in regimes:
        shapes += [{**base, key: v}, {**base, key: v + 1}]
    shapes.append({**base, size: 2 * t + 1})
    for t2 in tiles[1:]:
        shapes += [{**base, size: n} for n in (1, t2 - 1, t2 + 1, 2 * t2 + 1)]
    out = []
    for s in shapes:
        s = wl.clip({k: max(1, v) for k, v in s.items()})
        if s not in out and wl.in_domain(s):
            out.append(s)
    return out[:max_shapes]


# ---- parent side ------------------------------------------------------------------------------------------------
def _key(workload: str, old_src: str, new_src: str, shapes: list[dict]) -> str:
    h = hashlib.sha256()
    for part in (workload, old_src, new_src, json.dumps(shapes, sort_keys=True)):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _precheck(old_src: str, new_src: str) -> str | None:
    from ...backend.tirx import safety_scan
    for side, src in (("old", old_src), ("new", new_src)):
        problems = safety_scan(src)
        if problems:
            return f"refused to execute the {side} version: {'; '.join(problems)[:300]}"
    return None


def _run_child(workload: str, jobs: list[dict], timeout_s: float) -> dict[str, dict]:
    (BUILD / "tmp").mkdir(parents=True, exist_ok=True)
    total = timeout_s * max(1, len(jobs))
    with tempfile.TemporaryDirectory(dir=BUILD / "tmp", prefix="irdiff-") as tmp:
        job_f, out_f = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job_f.write_text(json.dumps({"workload": workload, "jobs": jobs, "job_timeout": timeout_s, "tmp": tmp}))
        cmd = [sys.executable, "-m", "akc.validation.changeaware.ir_worker", str(job_f), str(out_f)]
        err = ""
        try:
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=total,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            if p.returncode != 0:
                err = f"IR worker exited {p.returncode}: {(p.stderr or p.stdout)[-600:]}"
        except subprocess.TimeoutExpired:
            err = f"IR worker timed out after {total:.0f}s"
        done = json.loads(out_f.read_text()) if out_f.exists() else {}
    return {j["id"]: done.get(j["id"]) or {"ok": False, "error": err or "IR worker returned no result"} for j in jobs}


def analyze(workload: str, old_src: str, new_src: str, shapes: list[dict], timeout_s: float = TIMEOUT_S) -> dict:
    """IR change analysis of one revision (cached by the two sources and the shapes)."""
    key = _key(workload, old_src, new_src, shapes)
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    why = _precheck(old_src, new_src)
    res = {"ok": False, "error": why} if why else _run_child(workload, [
        {"id": key, "old": old_src, "new": new_src, "shapes": shapes}], timeout_s)[key]
    res.setdefault("wall_seconds", time.perf_counter() - t0)
    _CACHE[key] = res
    return res


def prefetch(workload: str, items: list[tuple[str, str, list[dict]]], timeout_s: float = TIMEOUT_S) -> None:
    """Analyze many revisions in ONE child process (tvm imported once, the shared old version traced once)."""
    jobs, seen = [], set()
    for old_src, new_src, shapes in items:
        key = _key(workload, old_src, new_src, shapes)
        if key in _CACHE or key in seen:
            continue
        why = _precheck(old_src, new_src)
        if why:
            _CACHE[key] = {"ok": False, "error": why}
            continue
        seen.add(key)
        jobs.append({"id": key, "old": old_src, "new": new_src, "shapes": shapes})
    if jobs:
        t0 = time.perf_counter()
        res = _run_child(workload, jobs, timeout_s)
        for k, r in res.items():
            r.setdefault("wall_seconds", (time.perf_counter() - t0) / len(jobs))
            _CACHE[k] = r


def clear_cache() -> None:
    _CACHE.clear()


# ---- child side -------------------------------------------------------------------------------------------------
class _Timeout(BaseException):          # not caught by the `except Exception` around user code
    pass


def _on_alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def _analyze_one(wl, job: dict, tmp: Path, mods: dict, flats: dict) -> dict:
    from ...backend.tirx import load_module, make_kernel
    from . import irdiff
    from .classify_ir import classify_ir

    def module(src: str):
        sha = hashlib.sha256(src.encode()).hexdigest()[:16]
        if sha not in mods:
            path = tmp / f"m_{sha}.py"
            path.write_text(src)
            try:
                mods[sha] = (load_module(path, f"akc_ir_{sha}"), str(path), None)
            except Exception as e:  # noqa: BLE001
                mods[sha] = (None, str(path), f"{type(e).__name__}: {str(e)[:300]}")
        return sha, mods[sha]

    def flat(sha, entry, shape):
        k = (sha, irdiff.shape_label(shape))
        if k not in flats:
            mod, path, err = entry
            if err:
                flats[k] = err
            else:
                try:
                    flats[k] = irdiff.flatten(make_kernel(mod, wl.meta, shape).func, path)
                except _Timeout:
                    raise
                except Exception as e:  # noqa: BLE001
                    flats[k] = f"{type(e).__name__}: {str(e)[:300]}"
        return flats[k]

    sa, ea = module(job["old"])
    sb, eb = module(job["new"])
    if eb[2]:
        return {"ok": False, "error": f"the new version does not load: {eb[2]}"}
    if ea[2]:
        return {"ok": False, "error": f"the old version does not load: {ea[2]}"}
    per_shape, skipped, traced, printer = [], {}, [], "script"
    for shape in job["shapes"]:
        lab = irdiff.shape_label(shape)
        fa, fb = flat(sa, ea, shape), flat(sb, eb, shape)
        if isinstance(fa, str):
            skipped[lab] = ("neither version traces: " if isinstance(fb, str) else "the old version does not trace: ") + fa
            continue
        if isinstance(fb, str):
            return {"ok": False, "error": f"the new version does not trace at {lab}: {fb}"}
        if "fallback" in (fa.printer, fb.printer):
            printer = "fallback"
        per_shape.append((lab, irdiff.diff_records(fa, fb)))
        traced.append(shape)
    if not traced:
        return {"ok": False, "error": "no shape could be traced: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())[:500]}
    merged = irdiff.merge(per_shape)
    labels = classify_ir(merged, len(traced))
    return {"ok": True, "error": None, "hunks": merged, "labels": [asdict(lab) for lab in labels], "shapes": traced,
            "skipped": skipped, "per_shape": {lab: len(h) for lab, h in per_shape}, "printer": printer}


def _child(job_path: str, out_path: str) -> None:
    j = json.loads(Path(job_path).read_text())
    out: dict[str, dict] = {}
    try:
        import tirx_kernels.tirx_lite  # noqa: F401  (tvm import; fail every job with one clear reason)
        from ...task.workload import load_workload
        wl = load_workload(j["workload"])
    except Exception as e:  # noqa: BLE001
        for job in j["jobs"]:
            out[job["id"]] = {"ok": False, "error": f"cannot trace tirx-lite here: {type(e).__name__}: {str(e)[:300]}"}
        Path(out_path).write_text(json.dumps(out))
        return
    mods, flats, tmp = {}, {}, Path(j["tmp"])
    has_alarm = hasattr(signal, "SIGALRM")
    for job in j["jobs"]:
        t0 = time.perf_counter()
        if has_alarm:
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(max(1, int(j.get("job_timeout", TIMEOUT_S))))
        try:
            res = _analyze_one(wl, job, tmp, mods, flats)
        except _Timeout:
            res = {"ok": False, "error": f"tracing timed out after {j.get('job_timeout', TIMEOUT_S)}s"}
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": f"IR analysis failed: {type(e).__name__}: {str(e)[:400]}"}
        finally:
            if has_alarm:
                signal.alarm(0)
        res["seconds"] = time.perf_counter() - t0
        out[job["id"]] = res
        Path(out_path).write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])
