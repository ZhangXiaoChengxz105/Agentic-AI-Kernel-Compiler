"""VoltaValidator: formal equivalence + race/deadlock checking of a candidate against the workload's reference
kernel in the same language, at the shapes given by the context (default: the workload's VOLTA default shape).

Input : Candidate (lang "cuda" or "tirx"), ctx.volta_shapes (list of shape dicts) or None
Output: ValidationReport("volta")
  per shape: EQUIV -> nothing; NOT_EQUIV/RACE/DEADLOCK/OUT_OF_BOUNDS/UNINIT/FOOTPRINT/DIV_BY_ZERO -> error finding;
  UNSUPPORTED/TIMEOUT -> info finding (verifier could not decide); CONFIG_ERROR/ERROR -> warning.
  decided = every shape ended in EQUIV or an error finding (used for "verification coverage").
  Each finding carries the shape, VOLTA's raw detail and an actionable hint (hints.py).
  artifacts: shapes, runs (per shape: status, wall_s, detail; tirx also: build seconds and the rewrites applied).
Limits (also stated in findings): real-number semantics (no rounding/overflow), CTA 0 only, exact sizes only.

cuda: candidate and the manifest's `references.cuda` are compiled with -DVOLTA (EXP -> __symexpf), layout from
      layouts.make_case.
tirx: candidate and `references.tirx` are traced at each shape (rows = layouts.ROWS), TIRx's generated CUDA is
      rewritten for VOLTA and compiled to PTX (volta/tirx.py: expf -> __symexpf, ex2.approx -> exp(t / log2 e),
      huge max identities -> +-inf, counted full-CTA barriers -> bar.sync id, compare-select -> max/min),
      layout from layouts.make_tirx_case. A trace / codegen failure at a VOLTA shape is a compile_error (the
      candidate does not build there); an nvcc failure on the rewritten source is a harness problem (warning).
"""
from __future__ import annotations

import time
from pathlib import Path

from ...backend.ptx import CompileError, compile_to_ptx
from ...paths import BUILD
from ...task.workload import Variant, load_workload
from ..interface import Candidate, Finding, ValidationContext, ValidationReport
from .hints import hint_for
from .layouts import FAIL_STATUSES, make_case, make_tirx_case, shape_label, volta_shape
from .runner import VoltaResult, run_analyze, run_compare
from .tirx import REWRITES, TirxBuildError, build_tirx_ptx

STATUS_MAP = {
    "NOT_EQUIV": ("not_equivalent", "error"), "RACE": ("race", "error"), "DEADLOCK": ("deadlock", "error"),
    "OUT_OF_BOUNDS": ("out_of_bounds", "error"), "UNINIT": ("uninitialized", "error"),
    "FOOTPRINT": ("footprint", "error"), "DIV_BY_ZERO": ("div_by_zero", "error"),
    "UNSUPPORTED": ("unsupported", "info"), "TIMEOUT": ("timeout", "info"),
    "CONFIG_ERROR": ("harness_error", "warning"), "ERROR": ("harness_error", "warning"),
}

_REF_CACHE: dict = {}          # (reference file, mtime, shape, rewrites) -> TirxPtx (the reference is fixed)


def result_finding(validator: str, r: VoltaResult, s: dict, lang: str = "cuda") -> Finding | None:
    """One VOLTA verdict at shape s -> None (EQUIV) or a Finding (STATUS_MAP; message, evidence and hint)."""
    if r.status == "EQUIV":
        return None
    kind, sev = STATUS_MAP.get(r.status, ("harness_error", "warning"))
    msg = (f"[{shape_label(s)}] VOLTA {r.status}"
           + (f": {r.mismatches} of {r.elements_checked or '?'} outputs proven different, e.g. "
              f"{', '.join(r.mismatch_examples[:3])}" if r.status == "NOT_EQUIV" else f": {r.detail}"))
    return Finding(kind, sev, validator, msg,
                   {"shape": s, "status": r.status, "which": r.which, "detail": r.detail,
                    "hint": hint_for(r, lang), "mismatches": r.mismatches})


class VoltaValidator:
    name = "volta"

    def __init__(self, timeout_s: float = 300.0, tirx_rewrites: tuple[str, ...] = REWRITES, build_root: Path | None = None):
        self.timeout_s = timeout_s
        self.build_root = build_root or BUILD / "volta_tirx"     # parallel workers each pass their own (no shared files)
        self.tirx_rewrites = tuple(tirx_rewrites)

    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport:
        t0 = time.perf_counter()
        wl = load_workload(cand.workload)
        if cand.lang == "tirx":
            return self._validate_tirx(cand, ctx, wl, t0)
        if cand.lang != "cuda":
            f = Finding("unsupported", "info", self.name, f"VOLTA path for {cand.lang!r} candidates is not implemented")
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
            f = result_finding(self.name, r, s)
            if f is not None:
                decided &= f.severity == "error"
                findings.append(f)
        return ValidationReport.build(self.name, findings, decided, t0, runs=runs, shapes=shapes)

    # ---- tirx-lite ------------------------------------------------------------------------------

    def _ref_build(self, ref: Variant, ref_mod, wl, vs: dict, out: Path):
        key = (str(ref.file), ref.file.stat().st_mtime_ns, tuple(sorted(vs.items())), self.tirx_rewrites)
        if key not in _REF_CACHE:
            _REF_CACHE[key] = build_tirx_ptx(ref_mod, wl.meta, vs, out / "_reference", ref.name, self.tirx_rewrites)
        return _REF_CACHE[key]

    def _validate_tirx(self, cand: Candidate, ctx: ValidationContext, wl, t0: float) -> ValidationReport:
        def info(msg: str) -> ValidationReport:
            return ValidationReport.build(self.name, [Finding("unsupported", "info", self.name, msg)], False, t0)

        if wl.volta is None or not wl.has_reference("tirx") or "tirx" not in wl.meta:
            return info("VOLTA is not set up for tirx-lite on this workload (needs a 'volta' block, a 'tirx' block "
                        "and references.tirx in its manifest)")
        try:
            import tirx_kernels.tirx_lite  # noqa: F401
            import tvm_tirx  # noqa: F401
        except ImportError as e:
            return info(f"tirx-lite is not installed here ({e}); VOLTA needs it to generate the kernel's CUDA")
        from ...backend.tirx import load_module
        label = cand.label or "cand"
        try:
            cand_mod = load_module(cand.materialize(), f"volta_cand_{label}")
        except PermissionError as e:
            f = Finding("compile_error", "error", self.name, f"module rejected by the safety scan: {e}")
            return ValidationReport.build(self.name, [f], True, t0)
        except Exception as e:  # noqa: BLE001  (the agent's module fails to import)
            f = Finding("compile_error", "error", self.name, f"module failed to import: {type(e).__name__}: {e}"[:1500])
            return ValidationReport.build(self.name, [f], True, t0)
        ref = wl.reference("tirx")
        ref_mod = load_module(ref.file, f"volta_ref_{ref.name}")
        out = self.build_root / cand.workload
        shapes = wl.cap_volta_shapes(ctx.volta_shapes or [wl.volta["default"]])
        findings, runs, decided = [], [], True
        for s in shapes:
            vs = volta_shape(cand.workload, s)
            run = {"shape": s}
            runs.append(run)
            try:
                rb = self._ref_build(ref, ref_mod, wl, vs, out)
            except TirxBuildError as e:
                run.update(status="ERROR", detail=f"reference build: {e}"[:300])
                findings.append(Finding("harness_error", "warning", self.name,
                                        f"[{shape_label(s)}] the tirx reference {ref.name} failed to build for "
                                        f"VOLTA ({e.stage})", {"shape": s, "stderr": e.message}))
                decided = False
                continue
            try:
                cb = build_tirx_ptx(cand_mod, wl.meta, vs, out / label, label, self.tirx_rewrites)
            except TirxBuildError as e:
                run.update(status=f"BUILD_{e.stage.upper()}", detail=e.message[:300])
                if e.stage in ("trace", "codegen"):
                    findings.append(Finding("compile_error", "error", self.name,
                                            f"[{shape_label(s)}] tirx-lite {e.stage} failed at this shape (VOLTA "
                                            f"build)", {"shape": s, "stderr": e.message}))
                elif e.stage == "entry":
                    findings.append(Finding("unsupported", "info", self.name, f"[{shape_label(s)}] {e.message}",
                                            {"shape": s}))
                    decided = False
                else:                                   # nvcc on the rewritten source: our problem, not the agent's
                    findings.append(Finding("harness_error", "warning", self.name,
                                            f"[{shape_label(s)}] nvcc failed on the VOLTA-rewritten tirx source",
                                            {"shape": s, "stderr": e.message}))
                    decided = False
                continue
            run.update(build_s=round(cb.seconds, 3), rewrites={"candidate": cb.rewrites, "reference": rb.rewrites},
                       dyn_smem={"candidate": cb.dyn_smem, "reference": rb.dyn_smem},
                       ptx={"candidate": str(cb.ptx), "reference": str(rb.ptx)})
            try:
                case = make_tirx_case(cand.workload, rb, cb, s, self.timeout_s)
            except ValueError as e:
                run.update(status="CONFIG_ERROR", detail=str(e)[:300])
                findings.append(Finding("harness_error", "warning", self.name, f"[{shape_label(s)}] {e}", {"shape": s}))
                decided = False
                continue
            a = None
            if 0 < cb.dyn_smem < case.dyn_shared:
                # compare gives both kernels the larger size: check the candidate alone at its own size first, so an
                # access past its shared memory is reported as OUT_OF_BOUNDS (not hidden or seen as UNINIT)
                a = run_analyze(case, cb.dyn_smem)
                run["alone"] = {"status": a.status, "wall_s": a.wall_s}
            r = a if a is not None and a.status in FAIL_STATUSES else run_compare(case)
            run.update(status=r.status, wall_s=r.wall_s, detail=r.detail[:300])
            f = result_finding(self.name, r, s, "tirx")
            if f is not None:
                decided &= f.severity == "error"
                findings.append(f)
        return ValidationReport.build(self.name, findings, decided, t0, runs=runs, shapes=shapes,
                                      rewrites=list(self.tirx_rewrites))
