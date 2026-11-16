"""Turn validation reports into the text the agent sees after each revision (its "observation").

render(reports, max_findings=8) -> str
  * errors first (definite bugs), each with: which validator, what failed, on which input, an example
    (index / got / want), the related change lines, and an actionable hint when one exists
  * then what could not be verified (info), so the model knows the limits of a "pass"
  * concise on purpose: GLM has no prompt-cache discount, so every resent token costs coins
Pilot evidence for why this matters: a raw VOLTA error string made the model rewrite a correct kernel
(introducing a bug); the same verdict with a one-line hint was fixed in one round.

render_perf(perf, best=None) -> str        speed of a candidate that passed: time, x PyTorch, roofline position
                                           (memory- or compute-bound, % of the GPU's peak, practical floor),
                                           and how it compares with the best correct version so far
render_observation(reports, perf=None, best=None) -> str   both, in the order the agent should read them
"""
from __future__ import annotations

from ..perf.report import PerfReport
from ..validation.interface import ValidationReport


def _example(ev: dict) -> str:
    ex = ev.get("examples") or []
    if not ex:
        return ""
    idx, got, want = ex[0]
    return f" e.g. output{idx}: got {got:.6g}, expected {want:.6g}"


def render(reports: list[ValidationReport], max_findings: int = 8) -> str:
    errors = [f for r in reports for f in r.findings if f.severity == "error"]
    warnings = [f for r in reports for f in r.findings if f.severity == "warning"]
    infos = [f for r in reports for f in r.findings if f.severity == "info"]
    lines = []
    if not errors:
        lines.append("All applicable checks passed (" + ", ".join(r.validator for r in reports) + ").")
    else:
        lines.append(f"Validation FAILED: {len(errors)} problem(s).")
        seen = set()
        for f in errors:
            key = (f.validator, f.kind, f.message[:80])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > max_findings:
                lines.append(f"... and {len(errors) - max_findings} more.")
                break
            lines.append(f"- [{f.validator}] {f.kind}: {f.message}{_example(f.evidence)}")
            if f.evidence.get("tag"):
                lines.append(f"  test purpose: {f.evidence['tag']}")
            for rc in f.evidence.get("related_change", [])[:2]:
                lines.append(f"  related change: {rc}")
            if f.evidence.get("hint"):
                lines.append(f"  hint: {f.evidence['hint']}")
            if f.evidence.get("stderr"):
                lines.append("  compiler output:\n" + f.evidence["stderr"][-1200:])
    for f in warnings[:3]:
        lines.append(f"- warning [{f.validator}] {f.message}")
    if infos:
        groups: dict[str, int] = {}
        for f in infos:                                   # collapse "[shape] same reason" repeats
            key = f"[{f.validator}] " + f.message.split("] ", 1)[-1][:140]
            groups[key] = groups.get(key, 0) + 1
        lines.append("Not verified: " + "; ".join(f"{k} (x{n})" if n > 1 else k for k, n in list(groups.items())[:4]))
    if any(r.validator == "volta" and r.passed and r.decided for r in reports):
        lines.append("Note: VOLTA proves equality over real numbers for the checked sizes only; rounding, overflow "
                     "and other sizes are covered by the tests.")
    return "\n".join(lines)


def _shape(shape: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in shape.items())


def render_perf(p: PerfReport, best: PerfReport | None = None, hints: bool = True) -> str:
    if not p.measured:
        return f"Speed not measured ({p.reason})."
    ridge, peak_bw, peak_fl = p.spec.get("ridge_flop_per_byte", 0), p.spec.get("dram_gbps", 0), p.spec.get("fp32_tflops", 0)
    head = f"Speed at {_shape(p.shape)} on {p.gpu} (median of {p.iters} launches, cold L2): {p.time_ms:.4f} ms"
    if p.speedup_vs_torch:
        head += f" = {p.speedup_vs_torch:.2f}x PyTorch's speed (PyTorch {p.torch_ms:.4f} ms; >1 means faster)"
    lines = [head + "."]
    if p.bound == "latency":
        lines.append("At this size the kernel runs for under 10 us: launch latency dominates, so the bandwidth and "
                     "FLOP/s numbers say little.")
    elif p.bound == "memory":
        lines.append(f"Memory-bound op ({p.intensity:.2f} FLOP/byte, below the GPU's ridge of {ridge:.1f}): effective "
                     f"bandwidth {p.achieved_gbps:.0f} GB/s = {p.pct_peak_bw:.0f}% of the {peak_bw:.0f} GB/s peak.")
        if p.copy_ms:
            lines.append(f"A plain device copy of the same {p.bytes / 1e6:.1f} MB takes {p.copy_ms:.4f} ms "
                         f"({p.bytes / (p.copy_ms * 1e-3) / 1e9:.0f} GB/s): that is the practical floor, "
                         f"{p.headroom:.2f}x faster than this kernel.")
        if hints:
            lines.append("Effective bandwidth counts each input read once and the output written once: extra passes "
                         "over global memory, uncoalesced or narrow (non-vectorized) accesses, or too few warps in "
                         "flight lower it.")
    else:
        msg = (f"Compute-bound op ({p.intensity:.0f} FLOP/byte, above the GPU's ridge of {ridge:.1f}): "
               f"{p.achieved_tflops:.2f} TFLOP/s = {p.pct_peak_flops:.0f}% of the {peak_fl:.1f} TFLOP/s FP32 peak")
        if p.torch_ms:
            t_tf = p.flops / (p.torch_ms * 1e-3) / 1e12
            msg += f"; PyTorch reaches {t_tf:.2f} TFLOP/s ({100 * t_tf / peak_fl:.0f}%)" if peak_fl else ""
        lines.append(msg + ".")
        if hints:
            lines.append("Typical levers: larger shared-memory tiles, register blocking (several outputs per thread), "
                         "vectorized loads, double buffering. Full fp32 accuracy is required (TF32 tensor cores fail "
                         "the tolerance).")
    if best is not None and best is not p and best.measured and best.time_ms:
        if p.faster_than(best):
            lines.append(f"New best correct version (previous best: {best.label or 'earlier'} at {best.time_ms:.4f} ms).")
        else:
            x = f", {best.speedup_vs_torch:.2f}x PyTorch" if best.speedup_vs_torch else ""
            lines.append(f"Best correct version so far: {best.label or 'earlier'} at {best.time_ms:.4f} ms{x}; this one "
                         f"is {p.time_ms / best.time_ms:.2f}x its time.")
    return "\n".join(lines)


def render_observation(reports: list[ValidationReport], perf: PerfReport | None = None,
                       best: PerfReport | None = None) -> str:
    """What the agent reads after one revision: validation first (correctness gates), then speed."""
    text = render(reports)
    if perf is not None:
        text += "\n\n" + render_perf(perf, best)
    return text
