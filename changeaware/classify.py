"""Task 2.2: classify a revision's changes into tiling / indexing / memory / sync / numerical.

classify(hunks, lang="cuda"|"tirx") -> list[Label]   (one Label per (category, rule, line), with evidence)

How rules are applied (same for both languages; only the vocabulary in LANGS differs)
  * pure additions / deletions: every line rule is checked against the whole line
  * modified lines: old and new lines are first paired by similarity; then
      - a line rule fires if the pair is about that construct (rule matches either side)
      - the math rule is stricter: it fires only if a math call itself differs (name or arguments),
        so editing a loop bound on a line that happens to contain fmaxf(...) is NOT "numerical"
      - pair rules look at the differing tokens: comparison operator, +1/-1, loop bound / trip count,
        guard, subscript (array index) expression
  * categories() ranks categories by amount of evidence

Languages
  cuda : CUDA C++ source; math calls and subscripts found textually
  tirx : tirx-lite Python, already normalized statement-by-statement by diff.normalize_python; math
         calls and subscripts come from the Python AST, so a shape list such as smem.alloc([bdy]) is
         not mistaken for an index expression

category   fires when the changed code ...
sync       adds/removes/moves barriers (__syncthreads / cta_sync / bar.sync), fences, mbarriers,
           async copies, atomics
tiling     touches tile/block constants, launch geometry, tile counts / loop trip counts, or rounds a
           size to a tile multiple
indexing   changes a comparison operator, a +1/-1, a loop bound, a guard, or a subscript expression
memory     touches shared/local allocations, vector/aliasing forms, global/shared load/store forms
numerical  changes a math call (exp/log/sqrt/rsqrt/max/min/fma/ex2...), infinities/eps constants, or a
           precision (half/bf16/double)
"""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Callable

from .diff import Hunk

CATEGORIES = ("tiling", "indexing", "memory", "sync", "numerical")

_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*[fF]?|<=|>=|==|!=|<<|>>|//|\+=|-=|\*=|/=|&&|\|\||[-+*/%<>=!&|^~?:;,.()\[\]{}]")
_COMPARISON = {"<", "<=", ">", ">=", "==", "!="}


@dataclass(frozen=True)
class Label:
    category: str
    rule: str
    line: int | None
    evidence: str


# ---- CUDA ---------------------------------------------------------------------------------------
_CUDA_MATH = re.compile(r"\b(EXP|expf?|__expf|exp2f?|logf?|__logf|log2f?|sqrtf?|rsqrtf?|__fdividef|fmaxf?|fminf?|fmaf?|tanhf?|powf?)\s*\(")
CUDA_RULES = [  # (category, rule name, regex)
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


def _balanced_calls(line: str, start_re: re.Pattern) -> list[str]:
    calls = []
    for m in start_re.finditer(line):
        depth, j = 0, m.end() - 1
        while j < len(line):
            depth += {"(": 1, ")": -1}.get(line[j], 0)
            if depth == 0:
                break
            j += 1
        calls.append(" ".join(line[m.start():j + 1].split()))
    return calls


def math_calls(line: str) -> list[str]:
    """CUDA: full text of every math call on a line, e.g. ['EXP(xr[i] - m)', 'fmaxf(m, xr[i])']."""
    return _balanced_calls(line, _CUDA_MATH)


def _cuda_subscripts(line: str) -> list[str]:
    return re.findall(r"\[([^\]]*)\]", line)


# ---- tirx-lite ----------------------------------------------------------------------------------
TIRX_MATH_NAMES = re.compile(r"^txl\.(exp2?|log2?|max|min|rsqrt|sqrt|fma|tanh|abs|ptx\.(ex2|lg2|rsqrt|sqrt|fma|max|min|tanh|rcp|div)\b.*)$")
TIRX_RULES = [
    ("sync", "barrier", r"\w*sync\w*\(|bar\.(sync|arrive)|MBarrier|mbarrier|fence|PipelineState|elect_sync"),
    ("sync", "async-copy", r"cp\.async|cp_async|\btma\b|wait_group|commit_group|TensorMap"),
    ("tiling", "launch-geometry", r"LaunchConfig|\bgrid\s*=|\bblock\s*=|\bBLOCK\s*=|\bTILE\s*=|\bNWARP\w*\s*="),
    ("tiling", "tile-size", r"\b(BLOCK|TILE|BM|BN|BK|NWARPS?|n_tiles|num_tiles|NTILE|iters)\b|ceildiv|//\s*(BLOCK|TILE)\b"),
    ("tiling", "trip-count", r"txl\.(serial|unroll)\("),
    ("memory", "shared/local alloc", r"smem_pool|\.alloc\(|alloc_local|alloc_tensor|swizzle|SwizzleMode"),
    ("memory", "load/store form", r"\b(ld|st)\.(shared|global_)\.\w*v[248]|reinterpret\("),
    ("numerical", "special-value", r"\beps\b|\binf\b|NEG_\w+|LOG2E|\d+(\.\d+)?e[-+]?\d+"),
    ("numerical", "precision", r"float16|bfloat16|\bhalf\b|\bf16\b|\bbf16\b|float64"),
]


def _parse_stmt(line: str):
    for src in (line, line + "\n    pass"):
        try:
            return ast.parse(src)
        except SyntaxError:
            continue
    return None


def _dotted(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return ""


def tirx_math_calls(line: str) -> list[str]:
    tree = _parse_stmt(line)
    if tree is None:
        return []
    return sorted(ast.unparse(n) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and TIRX_MATH_NAMES.match(_dotted(n.func)))


def _tirx_subscripts(line: str) -> list[str]:
    tree = _parse_stmt(line)
    if tree is None:
        return []
    return [ast.unparse(n.slice) for n in ast.walk(tree) if isinstance(n, ast.Subscript)] + \
           [ast.unparse(a) for n in ast.walk(tree) if isinstance(n, ast.Call) and _dotted(n.func).endswith("ptr_to")
            for a in n.args]


@dataclass(frozen=True)
class LangSpec:
    rules: list
    math_calls: Callable[[str], list[str]]
    subscripts: Callable[[str], list[str]]
    loop_bound: re.Pattern
    guard: re.Pattern


LANGS = {
    "cuda": LangSpec(CUDA_RULES, math_calls, _cuda_subscripts,
                     re.compile(r"\bfor\s*\(.*;.*[<>]=?.*;"), re.compile(r"\bif\s*\(.*[<>]=?")),
    "tirx": LangSpec(TIRX_RULES, tirx_math_calls, _tirx_subscripts,
                     re.compile(r"txl\.(serial|unroll)\(|\bfor\s+\w+\s+in\s+range\("), re.compile(r"txl\.if_\(.*[<>]=?|^if\s.*[<>]")),
}


# ---- shared machinery -------------------------------------------------------------------------
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


def _line_rules(spec: LangSpec, ln: int, text: str) -> list[Label]:
    out = [Label(c, r, ln, text) for c, r, rx in spec.rules if re.search(rx, text)]
    if spec.math_calls(text):
        out.append(Label("numerical", "math-call", ln, text))
    return out


def _loop_header(spec: LangSpec, line: str) -> str | None:
    m = spec.loop_bound.search(line)
    if not m:
        return None
    return line.split(";")[1] if ";" in line else line[m.start():]


def _pair_rules(spec: LangSpec, o_ln: int, old: str, new: str) -> list[Label]:
    ev = f"{old}  ->  {new}"
    labels = [Label(c, r, o_ln, ev) for c, r, rx in spec.rules if re.search(rx, old) or re.search(rx, new)]
    if spec.math_calls(old) != spec.math_calls(new):
        labels.append(Label("numerical", "math-call", o_ln, ev))
    rem, add = _token_delta(old, new)
    changed = set(rem) | set(add)
    if changed & _COMPARISON:
        labels.append(Label("indexing", "comparison-op", o_ln, ev))
    if "1" in changed and ("+" in changed or "-" in changed):
        labels.append(Label("indexing", "plus-minus-one", o_ln, ev))
    lo, ln_ = _loop_header(spec, old), _loop_header(spec, new)
    if lo is not None and ln_ is not None and lo != ln_:
        labels.append(Label("indexing", "loop-bound", o_ln, ev))
    if spec.guard.search(old) and spec.guard.search(new) and changed:
        labels.append(Label("indexing", "guard", o_ln, ev))
    so, sn = spec.subscripts(old), spec.subscripts(new)
    if so and len(so) == len(sn) and so != sn:
        labels.append(Label("indexing", "index-expr", o_ln, ev))
    return labels


def classify(hunks: list[Hunk], lang: str = "cuda") -> list[Label]:
    spec = LANGS[lang]
    labels: list[Label] = []
    for h in hunks:
        if h.kind == "mod":
            pairs, unpaired = _pair_lines(h.old, h.new)
            for (o_ln, o), (_, n) in pairs:
                labels.extend(_pair_rules(spec, o_ln, o, n))
            for ln, t in unpaired:
                labels.extend(_line_rules(spec, ln, t))
        else:
            for ln, t in [*h.old, *h.new]:
                labels.extend(_line_rules(spec, ln, t))
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
