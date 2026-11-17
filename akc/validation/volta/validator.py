"""VoltaValidator: formal equivalence + race/deadlock checking of a CUDA candidate against the workload's
reference kernel, at the shapes given by the context (default: the workload's VOLTA default shape).

Input : Candidate (lang "cuda"), ctx.volta_shapes (list of shape dicts) or None
Output: ValidationReport("volta")
  per shape: EQUIV -> nothing; NOT_EQUIV/RACE/DEADLOCK/OUT_OF_BOUNDS/UNINIT/FOOTPRINT/DIV_BY_ZERO -> error finding;
  UNSUPPORTED/TIMEOUT -> info finding (verifier could not decide); CONFIG_ERROR/ERROR -> warning.
  decided = every shape ended in EQUIV or an error finding (used for "verification coverage").
  Each finding carries the shape, VOLTA's raw detail and an actionable hint (hints.py).
Limits (also stated in findings): real-number semantics (no rounding/overflow), CTA 0 only, exact sizes only.
tirx-lite candidates: not wired up yet (Task 3) -> info. TIRx generates CUDA C++ that calls plain `expf`;
compiled to PTX, `expf` becomes `ex2.approx` + range reduction, which VOLTA does not read as exp. Our CUDA kernels
avoid this with EXP() -> `__symexpf` under -DVOLTA; the tirx path needs the same substitution on TIRx's generated
source (verified 2026-11-16: softmax_txl_ref's generated source contains `expf(...)`). Not yet tried on VOLTA.
"""
from __future__ import annotations

import time

from ...backend.ptx import CompileError, compile_to_ptx
from ...paths import BUILD
from ...task.workload import Variant, load_workload
from ..interface import Candidate, Finding, ValidationContext, ValidationReport
from .hints import hint_for
from .layouts import make_case, shape_label
from .runner import run_compare

STATUS_MAP = {
    "NOT_EQUIV": ("not_equivalent", "error"), "RACE": ("race", "error"), "DEADLOCK": ("deadlock", "error"),
    "OUT_OF_BOUNDS": ("out_of_bounds", "error"), "UNINIT": ("uninitialized", "error"),
    "FOOTPRINT": ("footprint", "error"), "DIV_BY_ZERO": ("div_by_zero", "error"),
    "UNSUPPORTED": ("unsupported", "info"), "TIMEOUT": ("timeout", "info"),
    "CONFIG_ERROR": ("harness_error", "warning"), "ERROR": ("harness_error", "warning"),
}


class VoltaValidator:
    name = "volta"

    def __init__(self, timeout_s: float = 300.0):
        self.timeout_s = timeout_s

    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport:
        t0 = time.perf_counter()
        wl = load_workload(cand.workload)
        if cand.lang != "cuda":
            f = Finding("unsupported", "info", self.name, "VOLTA path for tirx-lite candidates is not implemented "
                        "(TIRx's generated expf compiles to ex2.approx, which VOLTA does not read as exp)")
            return ValidationReport.build(self.name, [f], False, t0)
        if wl.volta is None or not wl.has_reference("cuda"):
            f = Finding("unsupported", "info", self.name, "VOLTA is not set up for this workload (needs a 'volta' "
                        "block and a CUDA reference in its manifest)")
            return ValidationReport.build(self.name, [f], False, t0)
        ref = wl.reference("cuda")
        out_dir = BUILD / "ptx" / cand.workload
        try:
            ref_ptx = compile_to_ptx(ref.file, out_dir / "ref", mode="volta").ptx
            b = compile_to_ptx(cand.materialize(), out_dir / (cand.label or "cand"), mode="volta")
        except CompileError as e:
            f = Finding("compile_error", "error", self.name, "nvcc failed (VOLTA build, -DVOLTA)", {"stderr": e.stderr[-2000:]})
            return ValidationReport.build(self.name, [f], True, t0)
        entry = cand.entry or (b.kernels[0] if len(b.kernels) == 1 else None)
        if entry not in b.kernels:
            f = Finding("compile_error", "error", self.name, f"kernel entry {cand.entry} not found; entries: {b.kernels}")
            return ValidationReport.build(self.name, [f], True, t0)
        v = Variant(entry, b.ptx, "cuda", "candidate", cand.block or tuple(wl.launch["block"]))
        shapes = wl.cap_volta_shapes(ctx.volta_shapes or [wl.volta["default"]])
        findings, runs, decided = [], [], True
        for s in shapes:
            r = run_compare(make_case(cand.workload, ref_ptx, ref, b.ptx, v, s, self.timeout_s))
            runs.append({"shape": s, "status": r.status, "wall_s": r.wall_s, "detail": r.detail[:300]})
            if r.status == "EQUIV":
                continue
            kind, sev = STATUS_MAP.get(r.status, ("harness_error", "warning"))
            decided &= sev == "error"
            msg = (f"[{shape_label(s)}] VOLTA {r.status}"
                   + (f": {r.mismatches} of {r.elements_checked or '?'} outputs proven different, e.g. "
                      f"{', '.join(r.mismatch_examples[:3])}" if r.status == "NOT_EQUIV" else f": {r.detail}"))
            findings.append(Finding(kind, sev, self.name, msg,
                                    {"shape": s, "status": r.status, "which": r.which, "detail": r.detail,
                                     "hint": hint_for(r), "mismatches": r.mismatches}))
        return ValidationReport.build(self.name, findings, decided, t0, runs=runs, shapes=shapes)
