"""Turn validation reports into the text the agent sees after each revision (its "observation").

render(reports, max_findings=8) -> str
  * errors first (definite bugs), each with: which validator, what failed, on which input, an example
    (index / got / want), the related change lines, and an actionable hint when one exists
  * then what could not be verified (info), so the model knows the limits of a "pass"
  * concise on purpose: GLM has no prompt-cache discount, so every resent token costs coins
Pilot evidence for why this matters: a raw VOLTA error string made the model rewrite a correct kernel
(introducing a bug); the same verdict with a one-line hint was fixed in one round.
"""
from __future__ import annotations

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
