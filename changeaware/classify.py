"""Task 2.2: classify a revision's changes into tiling / indexing / memory / sync / numerical.

classify(hunks) -> list[Label]   (one Label per (category, rule, line) that fired, with evidence)

How rules are applied
  * pure additions / deletions: every LINE_RULE is checked against the whole line
  * modified lines: old and new lines are first paired by similarity; then
      - a LINE_RULE fires if the pair is about that construct (rule matches either side) -- the line
        changed, so the construct it mentions is involved
      - the math rule is stricter: it fires only if a math call itself differs (name or arguments),
        so editing a loop bound on a line that happens to contain fmaxf(...) is NOT "numerical"
      - pair rules look at the differing tokens: comparison operator, +1/-1, loop bound, guard,
        index expression
  * a revision may get several categories; categories() ranks them by amount of evidence

category   fires when the changed code ...
sync       adds/removes/moves __syncthreads, __syncwarp, fences, barriers, cp.async waits, atomics
tiling     touches tile/block constants (#define TILE/BLOCK..., blockDim/gridDim), tile counts
           (ntiles, x / TILE), or rounds a size to a tile multiple ((N / B) * B)
indexing   changes a comparison operator (< vs <=), a +1/-1 adjustment, a loop bound, a guard
           condition, or an array index expression
memory     touches __shared__ declarations, vector types (float4...), __ldg, restrict/volatile,
           alignment, reinterpret_cast
numerical  changes a math call (exp/log/sqrt/rsqrt/max/min/fma/div...), infinities/eps constants,
           or the precision of a type (half/bf16/double)
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .diff import Hunk

CATEGORIES = ("tiling", "indexing", "memory", "sync", "numerical")

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*[fF]?|<=|>=|==|!=|<<|>>|\+=|-=|\*=|/=|&&|\|\||[-+*/%<>=!&|^~?:;,.()\[\]{}]")
_MATH_NAMES = r"EXP|expf?|__expf|exp2f?|logf?|__logf|log2f?|sqrtf?|rsqrtf?|__fdividef|fmaxf?|fminf?|fmaf?|tanhf?|powf?"
_MATH_START = re.compile(rf"\b({_MATH_NAMES})\s*\(")


@dataclass(frozen=True)
class Label:
    category: str
    rule: str
    line: int | None
    evidence: str


LINE_RULES = [  # (category, rule name, regex)
    ("sync", "barrier", r"__syncthreads|__syncwarp|bar\.sync|barrier\.sync|__threadfence|cuda::barrier|mbarrier"),
    ("sync", "async-copy", r"cp\.async|__pipeline_\w+|cp_async|commit_group|wait_group"),
    ("sync", "atomic", r"\batomic\w+\s*\("),
    ("tiling", "tile-constant", r"#\s*define\s+\w*(TILE|BLOCK|BM|BN|BK|WARPS?|THREADS|STAGES)\w*"),
    ("tiling", "launch-geometry", r"\bblockDim\b|\bgridDim\b|__launch_bounds__"),
    ("tiling", "tile-count", r"\bn?tiles?\b|\bnum_tiles\b|/\s*\(?\s*(TILE|BLOCK|BM|BN|BK)\b"),
    ("tiling", "round-to-tile", r"\(\s*\w+\s*/\s*\w+\s*\)\s*\*\s*\w+"),
    ("memory", "shared-decl", r"__shared__"),
    ("memory", "vector-type", r"\b(float|int|half|uint)[248]\b|reinterpret_cast|__ldg|__restrict__|volatile|__align__|alignas"),
    ("numerical", "special-value", r"INFINITY|FLT_MAX|FLT_MIN|CUDART_INF|\beps\b|\d+e-\d+"),
    ("numerical", "precision", r"\b(__half|half2?|__nv_bfloat16|bfloat16|double)\b|__float2half|__half2float"),
]
_COMPARISON = {"<", "<=", ">", ">=", "==", "!="}
_LOOP_BOUND = re.compile(r"\bfor\s*\(.*;.*[<>]=?.*;")
_GUARD = re.compile(r"\bif\s*\(.*[<>]=?")


def math_calls(line: str) -> list[str]:
    """Full text of every math call on a line, e.g. ['EXP(xr[i] - m)', 'fmaxf(m, xr[i])']."""
    calls = []
    for m in _MATH_START.finditer(line):
        depth, j = 0, m.end() - 1
        while j < len(line):
            depth += {"(": 1, ")": -1}.get(line[j], 0)
            if depth == 0:
                break
            j += 1
        calls.append(" ".join(line[m.start():j + 1].split()))
    return calls


def _token_delta(old: str, new: str) -> tuple[list[str], list[str]]:
    o, n = _TOKEN.findall(old), _TOKEN.findall(new)
    rem, add = list(o), list(n)
    for t in o:
        if t in add:
            add.remove(t)
            rem.remove(t)
    return rem, add


def _pair_lines(old: list, new: list) -> tuple[list, list]:
    """Greedy similarity pairing of old/new lines inside one hunk -> (pairs, unpaired lines)."""
    pairs, used = [], set()
    for n in new:
        best, best_r = None, 0.5
        for i, o in enumerate(old):
            if i not in used:
                r = difflib.SequenceMatcher(a=o[1], b=n[1]).ratio()
                if r > best_r:
                    best, best_r = i, r
        if best is None:
            pairs.append((None, n))
        else:
            used.add(best)
            pairs.append((old[best], n))
    unpaired = [o for i, o in enumerate(old) if i not in used] + [n for o, n in pairs if o is None]
    return [(o, n) for o, n in pairs if o is not None], unpaired


def _line_rules(ln: int, text: str) -> list[Label]:
    out = [Label(c, r, ln, text) for c, r, rx in LINE_RULES if re.search(rx, text)]
    if math_calls(text):
        out.append(Label("numerical", "math-call", ln, text))
    return out


def _pair_rules(o_ln: int, old: str, new: str) -> list[Label]:
    ev = f"{old}  ->  {new}"
    labels = [Label(c, r, o_ln, ev) for c, r, rx in LINE_RULES if re.search(rx, old) or re.search(rx, new)]
    if math_calls(old) != math_calls(new):
        labels.append(Label("numerical", "math-call", o_ln, ev))
    rem, add = _token_delta(old, new)
    changed = set(rem) | set(add)
    if changed & _COMPARISON:
        labels.append(Label("indexing", "comparison-op", o_ln, ev))
    if "1" in changed and ("+" in changed or "-" in changed):
        labels.append(Label("indexing", "plus-minus-one", o_ln, ev))
    if _LOOP_BOUND.search(old) and _LOOP_BOUND.search(new) and old.split(";")[1:2] != new.split(";")[1:2]:
        labels.append(Label("indexing", "loop-bound", o_ln, ev))
    if _GUARD.search(old) and _GUARD.search(new) and changed:
        labels.append(Label("indexing", "guard", o_ln, ev))
    idx_old, idx_new = re.findall(r"\[([^\]]*)\]", old), re.findall(r"\[([^\]]*)\]", new)
    if idx_old and len(idx_old) == len(idx_new) and idx_old != idx_new:
        labels.append(Label("indexing", "index-expr", o_ln, ev))
    return labels


def classify(hunks: list[Hunk]) -> list[Label]:
    labels: list[Label] = []
    for h in hunks:
        if h.kind == "mod":
            pairs, unpaired = _pair_lines(h.old, h.new)
            for (o_ln, o), (_, n) in pairs:
                labels.extend(_pair_rules(o_ln, o, n))
            for ln, t in unpaired:
                labels.extend(_line_rules(ln, t))
        else:
            for ln, t in [*h.old, *h.new]:
                labels.extend(_line_rules(ln, t))
    seen, out = set(), []
    for lab in labels:
        key = (lab.category, lab.rule, lab.line)
        if key not in seen:
            seen.add(key)
            out.append(lab)
    return out


def categories(labels: list[Label]) -> list[str]:
    """Distinct categories, most-evidenced first (ties broken by CATEGORIES order)."""
    counts = {c: sum(1 for lab in labels if lab.category == c) for c in CATEGORIES}
    return [c for c in sorted(CATEGORIES, key=lambda c: (-counts[c], CATEGORIES.index(c))) if counts[c]]
