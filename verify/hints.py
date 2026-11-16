"""Turn VOLTA verdicts into actionable, agent-readable feedback (seed of Task 3).

hint_for(result) -> str | None : advice for known verifier limitations / failure kinds, appended to the
raw verdict so the agent knows whether its kernel is wrong or the verifier needs a different spelling.
"""
from __future__ import annotations

from .volta import VoltaResult

_HINTS = [
    ("overflowed i128",
     "This is a verifier limitation, not necessarily a bug: VOLTA represents float constants as exact rationals "
     "with 128-bit integers, so constants with magnitude above ~1.7e38 (e.g. FLT_MAX, 3.4e38) overflow it. "
     "Use -INFINITY as the identity of max, or initialize the running max from a data element."),
    ("not a structured-CTA",
     "The verifier needs control flow and addresses that depend only on thread/block indices and sizes, "
     "never on data values. Replace data-dependent `if`s by fmaxf/fminf/selects."),
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


def hint_for(r: VoltaResult) -> str | None:
    text = f"{r.detail}\n{r.stderr}"
    for needle, hint in _HINTS:
        if needle in text:
            return hint
    if r.status == "NOT_EQUIV":
        return ("The verifier proved that some outputs differ from the reference for these exact sizes "
                "(real-number arithmetic, so this is a logic error, not rounding).")
    return None
