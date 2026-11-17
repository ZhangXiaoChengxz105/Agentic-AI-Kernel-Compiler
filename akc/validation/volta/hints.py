"""Turn VOLTA verdicts into actionable, agent-readable feedback (seed of Task 3).

hint_for(result, lang="cuda") -> str | None : advice for known verifier limitations / failure kinds, appended to the
raw verdict so the agent knows whether its kernel is wrong or the verifier needs a different spelling
(lang "tirx": the same advice in tirx-lite terms).
"""
from __future__ import annotations

from .runner import VoltaResult

_TRANSCENDENTAL = ("Verifier limitation: among transcendental functions it reads only exp, written as EXP(x) "
                   "(common.cuh); plain expf / __expf compile to ex2.approx, which it cannot read. lg2, sin, cos and "
                   "similar are out of its scope.")
_TRANSCENDENTAL_TIRX = ("Verifier limitation: among transcendental functions it reads only exp (txl.exp; "
                        "ex2.approx is read as exp(t / log2 e)). lg2, sin, cos and similar are out of its scope.")
_TRANSCENDENTAL_NEEDLES = ("transcendental", "Unsupported instruction: Ex2", "Unsupported instruction: Lg2",
                           "Unsupported instruction: Sin", "Unsupported instruction: Cos")

# checked in order: the specific limitations come before the generic "lowering failed"
_HINTS = [
    ("overflowed i128",
     "This is a verifier limitation, not necessarily a bug: VOLTA represents float constants as exact rationals "
     "with 128-bit integers, so constants with magnitude above ~1.7e38 (e.g. FLT_MAX, 3.4e38) overflow it. "
     "Use -INFINITY as the identity of max, or initialize the running max from a data element."),
    ("not a structured-CTA",
     "The verifier needs control flow and addresses that depend only on thread/block indices and sizes, "
     "never on data values. Replace data-dependent `if`s by fmaxf/fminf/selects."),
    ("non-finite constant",
     "Verifier limitation, not necessarily a bug: an infinite constant (e.g. the -inf identity of a max) reached an "
     "operation the verifier cannot simplify. It folds only max(-inf, x) = x and min(+inf, x) = x, so combine with "
     "max.f32 / min.f32 (txl.ptx[\"max.f32\"]) rather than comparisons or arithmetic on the identity."),
    *((needle, _TRANSCENDENTAL) for needle in _TRANSCENDENTAL_NEEDLES),
    ("thread-count operand",
     "Verifier limitation: it models only barriers over the whole CTA. Use bar.sync with the CTA's full thread "
     "count (or txl.cuda.cta_sync())."),
    ("register barrier id",
     "Verifier limitation: the barrier id must be a literal constant (e.g. txl.ptx.bar.sync(txl.uint32(0), ...))."),
    ("lowering failed",
     "The verifier could not parse/lower the PTX (an unsupported instruction or construct). "
     "Avoid inline PTX, cp.async, and other asynchronous primitives."),
    ("data race on",
     "Two threads access the same address between barriers and at least one writes: add the missing "
     "__syncthreads() (or make each thread touch only its own element)."),
    ("division by a symbolically zero value",
     "For these sizes an output divides by an expression that is identically zero, e.g. a sum over an empty "
     "range: check loop bounds/tile counts for sizes smaller than one tile or block."),
    ("VC too large",
     "Verifier limitation: the proof obligation grew too large for this size. Not necessarily a bug."),
    ("written footprints differ",
     "Your kernel writes a different set of output elements than the reference: check guards/bounds."),
]


# tirx-lite spelling of the same advice (the agent writes txl.* calls; inline PTX is normal there)
_TIRX_HINTS = {
    **{needle: _TRANSCENDENTAL_TIRX for needle in _TRANSCENDENTAL_NEEDLES},
    "not a structured-CTA":
        "The verifier needs control flow and addresses that depend only on thread/block indices and sizes, never on "
        "data values. Replace data-dependent txl.if_ by txl.max / txl.Select or max.f32.",
    "lowering failed":
        "The verifier could not lower an instruction of the generated PTX (named in the detail). Avoid cp.async, "
        "mbarrier and other asynchronous primitives; plain ld/st, shfl.sync, bar.sync and arithmetic are fine.",
    "data race on":
        "Two threads access the same address between barriers and at least one writes: add the missing "
        "txl.cuda.cta_sync() (or make each thread touch only its own element).",
}


def hint_for(r: VoltaResult, lang: str = "cuda") -> str | None:
    text = f"{r.detail}\n{r.stderr}"
    for needle, hint in _HINTS:
        if needle in text:
            return _TIRX_HINTS.get(needle, hint) if lang == "tirx" else hint
    if r.status == "NOT_EQUIV":
        return ("The verifier proved that some outputs differ from the reference for these exact sizes "
                "(real-number arithmetic, so this is a logic error, not rounding).")
    return None
