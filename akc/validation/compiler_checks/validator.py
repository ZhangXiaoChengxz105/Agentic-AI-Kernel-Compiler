"""CompilerChecksValidator: the shared "compiler checks" gate of configurations A / B / C (proposal: "Candidates that
pass CAKE's existing compiler checks will be checked by VOLTA"). tirx-lite is our CAKE substitute, so the gate is
tirx-lite's own checks plus the conventional sanitizer feedback the CAKE paper also uses.

Input : Candidate (lang "tirx"); ctx.device ("gpu" -> compute-sanitizer stage, "cpu" -> NumSim stage);
        ctx.extra["gate_shapes"] (override the shapes), ["gate_tools"] (subset of the four sanitizer tools),
        ["gate_parallel"] (default True: the four tools run at the same time)
Stages (artifacts["stages"]: seconds + status of each):
  1. static (tirx_static.py, one child process): safety scan, trace with the IR contract check, inspect_low_level_ir,
     check_well_formed, verify_memory, compile for sm_80, at every gate shape (tirx_static.gate_shapes: row ops
     rows=4 x N in {T+1, 1000} + rows=2 x N=v+1 per size regime; gemm 33x33x40 and 64x64x64).
     Any error -> compile_error (error), with the Python line when known; the dynamic stage is not run.
  2a. GPU: compute-sanitizer memcheck / racecheck / initcheck / synccheck (sanitizer.py) on cubins built from the
     generated CUDA. A report naming our kernel -> race / out_of_bounds / uninitialized / deadlock (error; racecheck
     INFO-level hazards -> info) with the generated-CUDA line as location; a kernel fault the tool did not explain ->
     runtime_error; a tool that crashed or timed out -> harness_error (warning). Outputs of the memcheck run are also
     compared with the float64 reference (artifacts["value_check"], information only: values are the tests' job).
  2b. CPU-only node: TIRx NumSim racecheck + synccheck at one small shape (tirx_static.numsim_shape). error -> error
     finding (race / deadlock / out_of_bounds / runtime_error), review -> warning, incomplete -> info "unsupported".
     The pipeline then tells the NumSim test runner not to run Racecheck again (ctx.extra["numsim_racecheck"]).
Output: ValidationReport("compiler_checks"); decided = the static stage finished and the dynamic stage reached a
        verdict (or a static error was found).
Legacy CUDA candidates are not part of A/B/C: one info finding, passed, not decided.
"""
from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...paths import BUILD, LOGS
from ...task.workload import load_workload
from ..interface import Candidate, Finding, ValidationContext, ValidationReport
from . import sanitizer as san
from .tirx_static import STAGES, gate_shapes, numsim_shape, run_static

NAME = "compiler_checks"
KIND_TEXT = {"race": "a data race on shared memory", "out_of_bounds": "an out-of-bounds or misaligned memory access",
             "uninitialized": "a read of uninitialized memory", "deadlock": "an invalid or divergent barrier",
             "runtime_error": "a kernel fault"}
_HEAD_NOISE = re.compile(r"\s+at\s+\S.*?\+0x[0-9a-fA-F]+\s+in\s+\S+?:\d+")


def _dims(shape: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in shape.items())


def _cuda_context(cu: Path, line: int) -> dict:
    """Where a line of TIRx's generated CUDA is: inside the kernel body, or in a TIRx helper (and which)."""
    try:
        lines = cu.read_text().splitlines()
    except OSError:
        return {"line": line}
    code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    body = next((i + 1 for i, ln in enumerate(lines) if re.search(r'extern\s+"C"\s+__global__.*\)\s*\{\s*$', ln)), None)
    ctx = {"file": str(cu), "line": line, "code": code[:240], "in_kernel": bool(body and line > body)}
    if not ctx["in_kernel"]:
        for i in range(min(line, len(lines)) - 1, -1, -1):
            m = re.search(r"__device__\s+[\w:<>\s\*&]+?\b(\w+)\s*\(", lines[i])
            if m:
                ctx["helper"] = m.group(1)
                break
    return ctx


def _report_locs(rep: dict, files: dict[str, dict]) -> list[dict]:
    """The generated-CUDA lines of one report, kernel-body lines first. A PC inside an inlined TIRx helper is mapped
    to its call site in the kernel through the cubin's line table (sanitizer.sass_line_map); without that, the
    reported (helper) line is kept and the helper is named."""
    found: list[dict] = []
    for pc in rep.get("pcs") or [{"file": f, "line": ln, "off": None, "access": None, "fn": ""} for f, ln in rep["locs"]]:
        rec = files.get(pc["file"])
        if rec is None:
            continue
        line, via = pc["line"], None
        call = (rec.get("sass") or {}).get(pc["off"]) if pc.get("off") is not None else None
        if call and call != line:
            line, via = call, pc["line"]
        c = {**_cuda_context(Path(rec["cuda"]), line), "shape": rec["shape"], "access": pc.get("access")}
        if via is not None:
            c["helper_line"] = via
            c["helper"] = (pc.get("fn") or "").split("(")[0] or None
        if not any(x["line"] == c["line"] and x.get("access") == c.get("access") for x in found):
            found.append(c)
    return sorted(found, key=lambda c: not c.get("in_kernel"))


def _cuda_location_text(c: dict, other: dict | None = None) -> str:
    def one(x):
        t = f"line {x['line']}" + (f" ({x['access']})" if x.get("access") else "")
        if x.get("helper") and not x.get("in_kernel"):
            t += f" inside TIRx helper {x['helper']}"
        return t
    return "generated CUDA " + one(c) + (f" and {one(other)}" if other else "")


class CompilerChecksValidator:
    name = NAME

    def __init__(self, tools: tuple[str, ...] = san.TOOLS, parallel: bool = True, timeout_s: float = san.RUN_TIMEOUT_S):
        self.tools, self.parallel, self.timeout_s = tuple(tools), parallel, timeout_s

    # ---------------------------------------------------------------------------------------------------------
    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport:
        t0 = time.perf_counter()
        if cand.lang != "tirx":
            f = Finding("skipped", "info", NAME, "compiler-checks gate not run: legacy CUDA is not part of configurations "
                        "A/B/C (the gate covers tirx-lite candidates)")
            return ValidationReport.build(NAME, [f], False, t0, stages=[])
        wl = load_workload(cand.workload)
        if "tirx" not in wl.meta:
            f = Finding("unsupported", "info", NAME, "the workload has no tirx-lite contract (manifest 'tirx' block)")
            return ValidationReport.build(NAME, [f], False, t0, stages=[])
        device = ctx.resolved_device()
        tools = tuple(ctx.extra.get("gate_tools") or self.tools)
        parallel = ctx.extra.get("gate_parallel", self.parallel)
        path = cand.materialize()
        digest = hashlib.sha1(cand.source.encode()).hexdigest()[:10]
        tag = f"{cand.label or 'cand'}_{digest}"
        out = BUILD / "gate" / cand.workload / tag
        shapes = [dict(s) for s in (ctx.extra.get("gate_shapes") or gate_shapes(cand.workload, cand.source))]
        stages: list[dict] = []
        art = {"device": device, "shapes": shapes, "tools": list(tools) if device == "gpu" else ["numsim:racecheck",
               "numsim:synccheck"], "stages": stages, "build_dir": str(out)}

        numsim_job = None
        if device != "gpu":
            ts = time.perf_counter()
            ns_shape = numsim_shape(cand.workload, cand.source)
            numsim_job = {"shape": ns_shape, "arrays": self._numsim_arrays(cand.workload, ns_shape, out / "numsim")}
            stages.append({"stage": "numsim_inputs", "seconds": round(time.perf_counter() - ts, 3), "status": "ok"})
        st = run_static(cand.workload, path, shapes, out / "cuda", cand.label or "cand", numsim=numsim_job)
        art["static"] = {k: st.get(k) for k in ("import_s", "load_s", "wall_s", "error")}
        art["static"]["shapes"] = [{"shape": r["shape"], "stages": r.get("stages", {}), "cuda": r.get("cuda"),
                                    "entry": r.get("entry"), "grid": r.get("grid"), "block": r.get("block"),
                                    "smem": r.get("smem"), "launch_error": r.get("launch_error")} for r in st["shapes"]]
        self._static_stages(st, stages)
        findings = self._static_findings(st, path)
        if findings:                                       # compile / static errors stop here
            return ValidationReport.build(NAME, findings, True, t0, **art)
        if st.get("error"):
            if "cannot run the tirx-lite checks here" in st["error"]:
                f = Finding("unsupported", "info", NAME, f"tirx-lite static checks not run: {st['error'][:300]}")
            else:
                f = Finding("harness_error", "warning", NAME, f"static checks did not finish: {st['error'][:400]}")
            return ValidationReport.build(NAME, [f], False, t0, **art)
        if device == "gpu":
            findings, decided = self._sanitizer(cand, st, tools, parallel, out, tag, stages, art)
        else:
            findings, decided = self._numsim_findings(st.get("numsim") or {}, path, stages, art)
        return ValidationReport.build(NAME, findings, decided, t0, **art)

    # ---- static ------------------------------------------------------------------------------------------------
    @staticmethod
    def _static_stages(st: dict, stages: list) -> None:
        stages.append({"stage": "safety_scan+import", "seconds": round((st.get("import_s") or 0) + (st.get("load_s") or 0), 3),
                       "status": "error" if any(e["stage"] in ("safety_scan", "import") for e in st["errors"]) else "ok"})
        for name in STAGES:
            recs = [r["stages"][name] for r in st["shapes"] if name in r.get("stages", {})]
            if recs:
                stages.append({"stage": name, "seconds": round(sum(r["seconds"] for r in recs), 4),
                               "status": "error" if any(r["status"] != "ok" for r in recs) else "ok", "shapes": len(recs)})
        stages.append({"stage": "static_child_wall", "seconds": round(st.get("wall_s") or 0, 3),
                       "status": "error" if st.get("error") else "ok"})

    @staticmethod
    def _static_findings(st: dict, path: Path) -> list[Finding]:
        groups: dict[tuple, dict] = {}
        for e in st["errors"]:
            text = e["message"] or ""
            detail = [ln.strip("- ").strip() for ln in text.splitlines() if ln.startswith("- ")]   # contract violations
            key = (e["stage"], "; ".join(detail)[:300] if detail else (text.splitlines()[0][:300] if text else ""))
            g = groups.setdefault(key, {"err": e, "shapes": [], "lines": []})
            if e.get("shape"):
                g["shapes"].append(e["shape"])
            g["lines"] += [ln for ln in e.get("lines", []) if ln not in g["lines"]]
        out = []
        what = {"safety_scan": "refused by the safety scan", "import": "the module failed to import",
                "trace": "tracing failed", "ir_contract": "the low-level IR contract is violated",
                "well_formed": "the traced IR is not well formed", "verify_memory": "illegal host-side memory access",
                "compile": "TIRx compilation for sm_80 failed", "timeout": "tracing / compiling hung"}
        for (stage, first), g in groups.items():
            dims = "; ".join(_dims(s) for s in g["shapes"][:3]) + (" ..." if len(g["shapes"]) > 3 else "")
            loc = f"line {g['lines'][0]}" if g["lines"] else None
            ev = {"stage": stage, "shapes": g["shapes"], "stderr": g["err"]["message"][-1500:], "python_lines": g["lines"],
                  "module": str(path)}
            msg = f"{'[' + dims + '] ' if dims else ''}{what.get(stage, stage)}: {first[:240]}"
            out.append(Finding("compile_error", "error", NAME, msg, ev, loc))
        return out

    # ---- GPU: compute-sanitizer --------------------------------------------------------------------------------
    def _sanitizer(self, cand, st, tools, parallel, out, tag, stages, art) -> tuple[list[Finding], bool]:
        findings: list[Finding] = []
        cs = san.find_compute_sanitizer()
        if cs is None:
            findings.append(Finding("harness_error", "warning", NAME, "compute-sanitizer not found (module load "
                                    "cudatoolkit/26.5_13.2): the sanitizer stage did not run"))
            return findings, False
        recs = [r for r in st["shapes"] if r.get("cuda")]
        for r in recs:
            if r.get("launch_error"):
                findings.append(Finding("unsupported", "info", NAME, f"[{_dims(r['shape'])}] cannot be launched under "
                                        f"compute-sanitizer: {r['launch_error'][:200]}", {"shape": r["shape"]}))
        recs = [r for r in recs if not r.get("launch_error")]
        if not recs:
            return findings, False
        ts = time.perf_counter()
        built, bad = {}, []

        def build(r):
            try:
                cubin = san.build_cubin(r["cuda"], out / "cubin")[0]
                return r["label"], cubin, san.sass_line_map(cubin, r.get("entry")), None
            except Exception as e:  # noqa: BLE001
                return r["label"], None, None, str(getattr(e, "stderr", e))[-800:]
        with ThreadPoolExecutor(len(recs)) as ex:
            for lab, cubin, sass, err in ex.map(build, recs):
                if err:
                    bad.append((lab, err))
                else:
                    built[lab] = cubin
                    next(r for r in recs if r["label"] == lab)["sass"] = sass
        stages.append({"stage": "nvcc_cubin", "seconds": round(time.perf_counter() - ts, 3),
                       "status": "error" if bad else "ok", "shapes": len(recs)})
        for lab, err in bad:                      # nvcc rejecting TIRx's own output: our harness, not the agent
            findings.append(Finding("harness_error", "warning", NAME, f"[{lab}] nvcc could not build TIRx's generated "
                                    "CUDA for the sanitizer", {"stderr": err}))
        recs = [r for r in recs if r["label"] in built]
        if not recs:
            return findings, False
        ts = time.perf_counter()
        units, golden = [], {}
        for i, r in enumerate(recs):
            args, gold, op, inputs = san.write_inputs(cand.workload, r["shape"], r["params"], out / "io" / r["label"], seed=i)
            golden[r["label"]] = (gold, op, inputs, args)
            units.append({"label": r["label"], "cubin": str(built[r["label"]]), "entry": r["entry"], "grid": r["grid"],
                          "block": r["block"], "smem": r.get("smem") or 0, "args": args})
        stages.append({"stage": "inputs", "seconds": round(time.perf_counter() - ts, 3), "status": "ok"})
        ts = time.perf_counter()
        log_dir = LOGS / "gate" / cand.workload / tag
        runs = san.run_tools(units, tools, out / "san", log_dir, cs, parallel, self.timeout_s)
        art["sanitizer_wall_s"] = round(time.perf_counter() - ts, 3)
        art["parallel"] = bool(parallel)
        files = {Path(r["cuda"]).name: r for r in recs}
        entry = recs[0]["entry"]
        decided = True
        per_tool = {}
        for tool in tools:
            run = runs[tool]
            parsed = san.parse_sanitizer(run["text"], entry, tool, files=set(files))
            child = run.get("child") or {}
            ok_run = run["exit"] != "timeout" and run.get("child") is not None and not child.get("error") and run["text"]
            stages.append({"stage": f"sanitizer:{tool}", "seconds": run["seconds"],
                           "status": "ok" if ok_run else "tool_failed", "log": run["log"]})
            tf = self._tool_findings(tool, parsed, files, run)
            faults = [u for u in child.get("units", []) if u.get("status") not in ("ok", "skipped")]
            if not ok_run:
                decided = False
                why = ("timed out" if run["exit"] == "timeout" else child.get("error") or
                       ("the launcher wrote no result" if run.get("child") is None else "no sanitizer log"))
                tf.append(Finding("harness_error", "warning", NAME, f"compute-sanitizer {tool} did not complete: {why}"
                                  + (" (a hang can also mean a deadlock)" if run["exit"] == "timeout" else ""),
                                  {"tool": tool, "log": run["log"], "tail": run.get("tail")}))
            elif faults and not any(f.severity == "error" for f in tf):
                u = faults[0]
                if u.get("kernel_fault"):
                    tf.append(Finding("runtime_error", "error", NAME, f"the kernel failed under compute-sanitizer {tool}: "
                                      f"{u.get('message', '')[:200]}", {"tool": tool, "unit": u, "log": run["log"]}))
                else:
                    decided = False
                    tf.append(Finding("harness_error", "warning", NAME, f"launcher error under {tool}: "
                                      f"{u.get('message', '')[:200]}", {"tool": tool, "unit": u, "log": run["log"]}))
            findings += tf
            own = [f for f in tf if f.severity == "error" and f.evidence.get("unit") is None]   # the tool's reports
            per_tool[tool] = {"caught": bool(own), "kinds": sorted({f.kind for f in own}),
                              "kernel_fault": any(f.kind == "runtime_error" and f.evidence.get("unit") for f in tf),
                              "n_reports": parsed["n_ours"], "by_severity": parsed["by_severity"],
                              "not_ours": parsed["not_ours"], "api_errors": parsed["api_errors"],
                              "seconds": run["seconds"], "exit": run["exit"], "log": run["log"],
                              "units": child.get("units", []), "completed": bool(ok_run)}
        art["sanitizer"] = per_tool
        art["logs"] = [runs[t]["log"] for t in tools]
        art["value_check"] = self._value_check(recs, golden)
        return findings, decided

    @staticmethod
    def _tool_findings(tool: str, parsed: dict, files: dict, run: dict) -> list[Finding]:
        groups: dict[tuple, dict] = {}
        for rep in parsed["reports"]:
            sev = "info" if rep["severity"] == "info" else "error"
            g = groups.setdefault((rep["kind"], sev), {"reps": [], "shapes": [], "locs": None})
            g["reps"].append(rep)
            for name, _ in rep["locs"]:
                s = files.get(name, {}).get("shape")
                if s and s not in g["shapes"]:
                    g["shapes"].append(s)
            if g["locs"] is None or not g["locs"][0].get("in_kernel"):
                locs = _report_locs(rep, files)
                if locs and (g["locs"] is None or locs[0].get("in_kernel")):
                    g["locs"] = locs
        out = []
        for (kind, sev), g in groups.items():
            reps = g["reps"]
            locs = g["locs"] or []
            c = locs[0] if locs else None
            other = next((x for x in locs[1:] if x["line"] != c["line"] and x.get("in_kernel")), None) if c else None
            heads = list(dict.fromkeys(_HEAD_NOISE.sub("", r["head"]) for r in reps))[:3]
            hz = sum(r["hazards"] or 0 for r in reps)
            dims = "; ".join(_dims(s) for s in g["shapes"][:3])
            msg = (f"{'[' + dims + '] ' if dims else ''}compute-sanitizer {tool} found {KIND_TEXT.get(kind, kind)}"
                   f"{' (INFO-level hazard only)' if sev == 'info' else ''}: {len(reps)} report(s)"
                   f"{f', {hz} hazards' if hz else ''}; e.g. {heads[0][:160] if heads else ''}")
            ev = {"tool": tool, "shapes": g["shapes"], "n_reports": len(reps), "hazards": hz or None, "heads": heads,
                  "log": run["log"], "python_lines": []}
            if c:
                ev["cuda"] = c
            if other:
                ev["cuda_other"] = other
            out.append(Finding(kind, sev, NAME, msg, ev, _cuda_location_text(c, other) if c else None))
        return out

    @staticmethod
    def _value_check(recs: list, golden: dict) -> list[dict]:
        """Outputs of the memcheck run vs the float64 reference (information: did the launch compute the op?)."""
        import numpy as np
        import torch

        from ..runners.check import compare
        res = []
        for r in recs:
            gold, op, inputs, args = golden[r["label"]]
            outs = [a for a in args if a.get("output")]
            try:
                got = np.load(outs[0]["save"])
                bits = torch.from_numpy(np.ascontiguousarray(got).view(np.int32).reshape(-1).copy())
                c = compare(bits, bits.view(torch.float32), gold, atol=op.atol, rtol=op.rtol,
                            scale=op.magnitude(inputs) if op.magnitude else None)
                res.append({"shape": r["shape"], "passed": c.passed, "reason": c.reason, "message": c.message[:200]})
            except Exception as e:  # noqa: BLE001  (e.g. the memcheck run stopped before saving)
                res.append({"shape": r["shape"], "passed": None, "reason": "not available", "message": str(e)[:200]})
        return res

    # ---- CPU: NumSim -------------------------------------------------------------------------------------------
    @staticmethod
    def _numsim_arrays(workload: str, shape: dict, out: Path) -> dict[str, str]:
        import numpy as np

        from ...task.reference import OPS
        op, wl = OPS[workload], load_workload(workload)
        inputs = op.make_inputs(shape, "normal", 0)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for n in wl.meta["tirx"]["tensors"]:
            f = out / f"{n}.npy"
            arr = inputs[n].numpy() if n in inputs else np.zeros(op.out_shape(shape), np.float32)
            np.save(f, np.ascontiguousarray(arr.astype(np.float32)))
            arrays[n] = str(f)
        return arrays

    @staticmethod
    def _numsim_kind(checker: str, kind: str, message: str) -> str:
        t = f"{kind} {message}".lower()
        if "race" in t:
            return "race"
        if "out-of-bounds" in t or "out_of_bounds" in t or "out of bounds" in t or "oob" in kind.lower():
            return "out_of_bounds"
        if "uninit" in t:
            return "uninitialized"
        if checker == "synccheck" or "barrier" in t or "deadlock" in t or "sync" in kind.lower():
            return "deadlock"
        return "runtime_error"

    def _numsim_findings(self, ns: dict, path: Path, stages: list, art: dict) -> tuple[list[Finding], bool]:
        art["numsim"] = ns
        findings, decided = [], True
        if ns.get("error"):
            stages.append({"stage": "numsim", "seconds": ns.get("seconds", 0), "status": "tool_failed"})
            return [Finding("harness_error", "warning", NAME, f"NumSim checks did not run: {ns['error'][:300]}")], False
        shape = ns.get("shape") or {}
        for checker in ("racecheck", "synccheck"):
            r = ns.get(checker)
            if r is None:
                decided = False
                continue
            stages.append({"stage": f"numsim:{checker}", "seconds": r.get("seconds", 0),
                           "status": "tool_failed" if r["verdict"] == "crash" else r["verdict"]})
            if r["verdict"] == "crash":
                if "out-of-bounds" in r.get("error", "") or "out of bounds" in r.get("error", ""):
                    findings.append(Finding("out_of_bounds", "error", NAME, f"[{_dims(shape)}] NumSim {checker}: "
                                            f"{r['error'][:240]}", {"tool": f"numsim:{checker}", "shape": shape,
                                                                     "python_lines": r.get("lines", [])},
                                            f"line {r['lines'][0]}" if r.get("lines") else None))
                else:
                    decided = False
                    findings.append(Finding("harness_error", "warning", NAME, f"NumSim {checker} crashed: "
                                            f"{r.get('error', '')[:240]}", {"tool": f"numsim:{checker}"}))
                continue
            if r["verdict"] == "incomplete" and not any(x["status"] == "error" for x in r.get("findings", [])):
                decided = False                            # NumSim could not cover the run: no verdict
            seen = set()
            for f in r.get("findings", []):
                if f["status"] == "incomplete":
                    kind, sev = "unsupported", "info"
                else:
                    kind, sev = self._numsim_kind(checker, f["kind"], f["message"]), \
                        ("error" if f["status"] == "error" else "warning")
                key = (kind, sev, tuple(f.get("lines", [])))
                if key in seen:
                    continue
                seen.add(key)
                lines, callers = f.get("lines", []), f.get("callers", [])
                loc = None
                if lines:
                    loc = ("lines " + " and ".join(str(x) for x in lines[:2]) if len(lines) > 1 else f"line {lines[0]}")
                    loc += f" (in a helper called from line {callers[0]})" if callers else ""
                findings.append(Finding(kind, sev, NAME, f"[{_dims(shape)}] NumSim {checker} {f['status']}: "
                                        f"{f['kind']}: {(f.get('summary') or f['message'])[:300]}",
                                        {"tool": f"numsim:{checker}", "shape": shape, "python_lines": lines,
                                         "call_sites": callers, "numsim_kind": f["kind"], "detail": f["message"][:600],
                                         "module": str(path)}, loc))
        art["sanitizer"] = {f"numsim:{c}": {"caught": any(f.severity == "error" and f.evidence.get("tool") == f"numsim:{c}"
                                                          for f in findings),
                                            "verdict": (ns.get(c) or {}).get("verdict"),
                                            "seconds": (ns.get(c) or {}).get("seconds")}
                            for c in ("racecheck", "synccheck")}
        return findings, decided
