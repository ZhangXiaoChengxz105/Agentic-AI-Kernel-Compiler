"""Test executors shared by the fixed-test and change-aware validators.

run(cand, tests, device) -> list[outcome dict]     picks the executor:
    cuda + gpu  -> runners.gpu (driver-API launch of the candidate's PTX)
    tirx + gpu  -> runners.gpu (TIRx compile per shape)
    tirx + cpu  -> runners.numsim (CPU simulation + Racecheck; numsim_racecheck=False skips Racecheck when the
                   compiler-checks gate already ran it)
    cuda + cpu  -> nothing can execute: every test returns "skipped" (info)
outcome_to_finding(outcome, validator) -> Finding | None   (None = test passed)
build_for_gpu(cand) -> (kernel file, CUDA entry | None, block)   raises BuildError (also used by perf.measure)
"""
from __future__ import annotations

from ..interface import Candidate, Finding
from .check import CheckResult

# harness reason -> (Finding.kind, severity)
REASON_MAP = {
    "canary": ("out_of_bounds", "error"),
    "out_of_bounds": ("out_of_bounds", "error"),
    "unwritten": ("unwritten", "error"),
    "nonfinite": ("nonfinite", "error"),
    "mismatch": ("mismatch", "error"),
    "nondeterministic": ("nondeterministic", "error"),
    "race": ("race", "error"),
    "cuda_error": ("runtime_error", "error"),
    "runtime_error": ("runtime_error", "error"),
    "compile_error": ("compile_error", "error"),
    "skipped": ("skipped", "info"),
}


def _skip_all(tests, why: str) -> list[dict]:
    return [{**t.to_dict(), **CheckResult(False, "skipped", message=why).to_dict()} for t in tests]


class BuildError(Exception):
    """The candidate cannot be turned into something the GPU executor loads (message = compiler output)."""


def build_for_gpu(cand: Candidate) -> tuple:
    """Candidate -> (file the GPU executor loads, CUDA entry name or None, launch block).
    CUDA: compiled to PTX (backend.ptx, mode "gpu"); tirx: the module file itself (compiled per shape later)."""
    from ...task.workload import load_workload
    wl = load_workload(cand.workload)
    block = cand.block or tuple(wl.launch["block"])
    if cand.lang != "cuda":
        return cand.materialize(), None, block
    from ...backend.ptx import CompileError, compile_to_ptx
    from ...paths import BUILD
    try:
        b = compile_to_ptx(cand.materialize(), BUILD / "ptx_gpu" / (cand.label or "cand"), mode="gpu")
    except CompileError as e:
        raise BuildError(e.stderr[-1500:]) from e
    entry = cand.entry or (b.kernels[0] if len(b.kernels) == 1 else None)
    if entry not in b.kernels:
        raise BuildError(f"cannot find kernel entry {cand.entry}; entries: {b.kernels}")
    return b.ptx, entry, block


def run(cand: Candidate, tests: list, device: str, numsim_racecheck: bool = True) -> list[dict]:
    if not tests:
        return []
    if device != "gpu":
        if cand.lang == "cuda":
            return _skip_all(tests, "no GPU here: dynamic tests not executed (run on a GPU node)")
        from .numsim import run_tests_numsim_isolated
        return run_tests_numsim_isolated(cand.workload, cand.materialize(), tests, racecheck=numsim_racecheck)
    try:
        kernel_file, entry, block = build_for_gpu(cand)
    except BuildError as e:
        return [{**t.to_dict(), **CheckResult(False, "compile_error", message=str(e)).to_dict()} for t in tests]
    from .gpu import run_tests_isolated
    return run_tests_isolated(cand.workload, cand.lang, kernel_file, entry, block, tests)


def outcome_to_finding(o: dict, validator: str) -> Finding | None:
    if o.get("passed"):
        return None
    kind, sev = REASON_MAP.get(o.get("reason", ""), ("harness_error", "warning"))
    dims = ",".join(f"{k}={v}" for k, v in o.get("shape", {}).items())
    ev = {k: o.get(k) for k in ("shape", "dist", "seed", "repeat", "misalign", "tag", "n_bad", "n", "max_abs_err",
                                "max_rel_err", "examples") if o.get(k) not in (None, [], 0, "")}
    return Finding(kind, sev, validator, f"[{dims} {o.get('dist', '')}] {o.get('message', '')}".strip(), ev)
