"""Kernel speed: the agent's objective (how fast, how far from the hardware limit), NOT a validator.

    measure(candidate, shape=None) -> PerfReport          GPU only; correctness re-checked at the benchmark shape
    analyze(workload, shape, timing, spec, ...) -> PerfReport   roofline arithmetic (CPU-testable)
    feedback.render.render_perf(report, best=None) -> str  the text the agent sees

Validation decides pass/fail and differs between configurations A/B/C; speed has no pass/fail, is measured
only for candidates that passed, and is measured identically in A/B/C.
"""
from .hardware import SPECS, GpuSpec, spec_for
from .measure import measure
from .report import PerfReport, analyze

__all__ = ["GpuSpec", "PerfReport", "SPECS", "analyze", "measure", "spec_for"]
