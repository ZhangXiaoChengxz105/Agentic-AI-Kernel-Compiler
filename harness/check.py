"""Task 1.5 core: compare a kernel output with the float64 golden result.

Guards against false passes and silent memory bugs:
  * sentinel : the output buffer is pre-filled with a NaN carrying a special payload (bits 0x7FC0DEAD);
               any element still holding it afterwards was never written ("unwritten")
  * canaries : CANARY elements before and after the output, also sentinel-filled; any change means
               the kernel wrote out of bounds ("canary")
  * finite   : NaN/Inf where the golden value is finite ("nonfinite")
  * tolerance: |got - want| <= atol + rtol * scale, scale = |want| (or |A|@|B| for gemm)
Reason priority: canary > unwritten > nonfinite > mismatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

SENTINEL_BITS = 0x7FC0DEAD
CANARY = 256


@dataclass
class CheckResult:
    passed: bool
    reason: str                       # ok | canary | unwritten | nonfinite | mismatch | nondeterministic | cuda_error | skipped
    n: int = 0
    n_bad: int = 0
    n_nonfinite: int = 0
    n_unwritten: int = 0
    canary_ok: bool = True
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    examples: list = field(default_factory=list)   # up to 5 (index, got, want)
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def alloc_output(numel: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (whole buffer incl. canaries, float32 view of the logical output)."""
    buf = torch.full((numel + 2 * CANARY,), SENTINEL_BITS, dtype=torch.int32, device=device)
    return buf, buf[CANARY:CANARY + numel].view(torch.float32)


def canary_intact(buf: torch.Tensor, numel: int) -> bool:
    head, tail = buf[:CANARY], buf[CANARY + numel:]
    return bool((head == SENTINEL_BITS).all() and (tail == SENTINEL_BITS).all())


def unwritten_mask(out: torch.Tensor) -> torch.Tensor:
    return out.view(torch.int32) == SENTINEL_BITS


def _examples(mask: torch.Tensor, got: torch.Tensor, want: torch.Tensor, shape) -> list:
    """Up to 5 (index, got, want) triples where mask is true, for human/agent-readable feedback."""
    out = []
    for ix in mask.nonzero()[:5].tolist():
        t = tuple(ix)
        out.append((ix, float(got[t]), float(want[t])))
    return out


def compare(buf: torch.Tensor, out: torch.Tensor, golden: torch.Tensor, *, atol: float, rtol: float,
            scale: torch.Tensor | None = None) -> CheckResult:
    shape = golden.shape
    got = out.reshape(shape).double().cpu()
    raw = out.reshape(shape).cpu()
    want = golden.double().cpu()
    n = want.numel()

    canary_ok = canary_intact(buf.cpu(), out.numel())
    unwritten = unwritten_mask(raw)
    nonfinite = (~torch.isfinite(got)) & torch.isfinite(want) & ~unwritten
    ref_scale = (scale.double().cpu() if scale is not None else want.abs())
    tol = atol + rtol * ref_scale
    diff = (got - want).abs()
    bad = ~(diff <= tol)                      # NaN-safe: NaN diff counts as bad
    finite_both = torch.isfinite(got) & torch.isfinite(want)
    max_abs = float(diff[finite_both].max()) if finite_both.any() else float("inf")
    rel = diff / want.abs().clamp_min(1e-30)
    max_rel = float(rel[finite_both].max()) if finite_both.any() else float("inf")

    r = CheckResult(passed=False, reason="ok", n=n, n_bad=int(bad.sum()), n_nonfinite=int(nonfinite.sum()),
                    n_unwritten=int(unwritten.sum()), canary_ok=canary_ok, max_abs_err=max_abs, max_rel_err=max_rel)
    if not canary_ok:
        r.reason, r.message = "canary", "kernel wrote outside the output buffer (out-of-bounds store)"
    elif r.n_unwritten:
        r.reason, r.message = "unwritten", f"{r.n_unwritten} of {n} output elements were never written"
        r.examples = _examples(unwritten, got, want, shape)
    elif r.n_nonfinite:
        r.reason, r.message = "nonfinite", f"{r.n_nonfinite} outputs are NaN/Inf where the reference is finite"
        r.examples = _examples(nonfinite, got, want, shape)
    elif r.n_bad:
        r.reason, r.message = "mismatch", (f"{r.n_bad} of {n} outputs exceed tolerance "
                                           f"(max abs err {max_abs:.3g}, max rel err {max_rel:.3g})")
        r.examples = _examples(bad, got, want, shape)
    else:
        r.passed, r.message = True, f"all {n} outputs within tolerance (max abs err {max_abs:.3g})"
    return r
