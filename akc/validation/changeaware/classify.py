"""Task 2.2: classify a revision's changes into tiling / indexing / memory / sync / numerical.

classify(hunks, lang="cuda"|"tirx") -> list[Label]   (one Label per (category, rule, line), with evidence)
proposal_category(cat)                               indexing -> tiling: the proposal's 4 categories, for reports
(tirx-lite revisions are normally classified on the IR, classify_ir.py; this text / AST classifier is the
CUDA path and the tirx fallback when tracing fails)

How rules are applied (same for both languages; only the vocabulary in LANGS differs)
  * pure additions / deletions: every line rule is checked against the whole line
  * modified lines: old and new lines are first paired by similarity; then
      - a line rule fires if the pair is about that construct (rule matches either side)
      - the math rule is stricter: it fires only if a math call itself differs (name or arguments),
        so editing a loop bound on a line that happens to contain fmaxf(...) is NOT "numerical"
      - pair rules look at the differing tokens: comparison operator, +1/-1, loop bound / trip count,
        guard, subscript (array index) expression
      - loop / if headers are split off a line first: a changed loop condition is "loop-bound", a changed loop
        start or step ("s >>= 1" -> "s >>= 2") "loop-step", a changed if-condition "guard"; the statement after
        the headers is compared with array subscripts removed. If no other rule fired on the pair: assigning a
        float (a float variable, an element of a float array / pointer, a float store) makes it numerical
        "arith", anything else (index / pointer arithmetic) indexing "index-arith"
  * classify(..., src=new source) uses the source to know which names are floats
  * categories() ranks categories by amount of evidence

Languages
  cuda : CUDA C++ source; math calls and subscripts found textually
  tirx : tirx-lite Python, already normalized statement-by-statement by diff.normalize_python; math
         calls and subscripts come from the Python AST, so a shape list such as smem.alloc([bdy]) is
         not mistaken for an index expression

category   fires when the changed code ...
sync       adds/removes/moves barriers (__syncthreads / cta_sync / bar.sync), fences, mbarriers,
           async copies, atomics, warp shuffles / votes
tiling     touches tile/block constants, launch geometry, tile counts / loop trip counts, or rounds a
           size to a tile multiple
indexing   changes a comparison operator, a +1/-1, a loop bound / start / step, a guard, a subscript
           expression, or integer index / pointer arithmetic
memory     touches shared/local allocations, vector/aliasing forms, global/shared load/store forms
numerical  changes a math call (exp/log/sqrt/rsqrt/max/min/fma/ex2...), float arithmetic, infinities/eps
           constants, or a precision (half/bf16/double)
"""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Callable

from .diff import Hunk

CATEGORIES = ("tiling", "indexing", "memory", "sync", "numerical")
# The proposal names 4 categories (tiling, memory access, synchronization, numerical); indexing is kept separate
# inside the tool (its tests differ a little: rows > 1 so an off-by-one read hits real data) but reported as tiling.
PROPOSAL_CATEGORIES = ("tiling", "memory", "sync", "numerical")


def proposal_category(cat: str) -> str:
    """Report view of a category: indexing -> tiling; the other four are the proposal's own."""
    return "tiling" if cat == "indexing" else cat

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
    ("sync", "warp-shuffle", r"__shfl\w*|__ballot_sync|__any_sync|__all_sync|__match_\w+_sync|__reduce_\w+_sync|__activemask|cooperative_groups|\bcg::"),
    ("tiling", "tile-constant", r"#\s*define\s+\w*(TILE|BLOCK|BM|BN|BK|WARPS?|THREADS|STAGES|TPB|NWARP|UNROLL|VEC)\w*"),
    ("tiling", "compile-constant", r"#\s*define\s+\w+\s+\(?\s*\d+|\bconstexpr\s+(int|unsigned)\s+\w+\s*=\s*\d+"),
    ("memory", "smem-size", r"#\s*define\s+\w*(SMEM|SHARED|SHM)\w*"),
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
    ("sync", "warp-shuffle", r"shfl|warp_reduce|redux|ballot|activemask"),
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
    calls = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and TIRX_MATH_NAMES.match(_dotted(n.func))]
    refs = [_dotted(n) for n in ast.walk(tree)                      # passed as a value: reduce(m, txl.max)
            if isinstance(n, ast.Attribute) and TIRX_MATH_NAMES.match(_dotted(n))]
    return sorted(calls + refs)


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
    split: Callable[[str], tuple[list[tuple[str, str, str]], str]]     # line -> (headers, statement)
    lhs: Callable[[str], str | None]                                   # statement -> assigned name ("" = float store)
    float_names: Callable[[str], set[str]]                             # source -> names holding floats


def _balanced(text: str, open_at: int) -> int:
    """Index just past the parenthesis group starting at text[open_at] == '('."""
    depth = 0
    for j in range(open_at, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return j + 1
    return len(text)


_CUDA_HEAD = re.compile(r"^\s*(?:\}\s*)?(?:else\s+)?(for|while|if)\s*\(")


def _cuda_split(line: str):
    """'for (int i = t; i < N; i += B) if (c) y[i] = v;' -> ([('for', 'i < N', header), ('if', 'c', header)], 'y[i] = v;')"""
    heads, rest = [], line
    while True:
        m = _CUDA_HEAD.match(rest)
        if not m:
            return heads, rest.strip()
        end = _balanced(rest, m.end() - 1)
        inner = rest[m.end():end - 1]
        cond = inner.split(";")[1].strip() if m.group(1) == "for" and inner.count(";") >= 2 else inner.strip()
        heads.append((m.group(1), cond, inner.strip()))
        rest = rest[end:].lstrip(" {")


def _tirx_split(line: str):
    t = line.strip()
    m = re.search(r"txl\.if_\(", t)
    if m:
        end = _balanced(t, m.end() - 1)
        return [("if", t[m.end():end - 1].strip(), t)], ""
    m = re.match(r"(if|elif|while)\s+(.*):$", t)
    if m:
        return [("if" if m.group(1) != "while" else "while", m.group(2), t)], ""
    if re.match(r"(for\s|with\s+txl\.(serial|unroll)\()", t):
        return [("for", t, t)], ""
    return [], t


_CUDA_ASSIGN = re.compile(r"(?<![=!<>+\-*/&|^%])(=|\+=|-=|\*=|/=)(?!=)")


def _cuda_lhs(stmt: str) -> str | None:
    """Assigned name; 'name[]' when an element is assigned (y[i] = ...)."""
    m = _CUDA_ASSIGN.search(stmt)
    if not m:
        return None
    names = re.findall(r"(\w+)\s*(\[[^\]]*\])?\s*$", stmt[:m.start()])
    return (names[0][0] + ("[]" if names[0][1] else "")) if names else None


def _tirx_lhs(stmt: str) -> str | None:
    if re.search(r"\.st\.\w+\.f(16|32|64)\(", stmt):
        return ""                                        # float store
    m = re.match(r"txl\.assign\(\s*(\w+)\s*,", stmt) or re.match(r"(\w+)\s*=(?!=)", stmt)
    return m.group(1) if m else None


def _cuda_floats(src: str) -> set[str]:
    """Float scalars ('m') and float arrays / pointers, whose ELEMENTS are floats ('red[]', 'y[]')."""
    t = r"\b(?:float|double|half|__half)"
    scalars = set(re.findall(t + r"\s+(\w+)\b(?!\s*\[)", src))
    arrays = set(re.findall(t + r"\s+(\w+)\s*\[", src)) | \
        set(re.findall(t + r"\s*\*\s*(?:const\s+)?(?:__restrict__\s+)?(\w+)", src))
    return scalars | {a + "[]" for a in arrays}


def _tirx_floats(src: str) -> set[str]:
    return set(re.findall(r"^\s*(\w+)\s*=\s*txl\.(?:local_scalar|float32|float16)\(\s*(?:\"float|NEG|[-\d])", src, re.M)) | \
        set(re.findall(r"^\s*(\w+)\s*=\s*txl\.smem_pool\(\)\.alloc\([^)]*float", src, re.M))


LANGS = {
    "cuda": LangSpec(CUDA_RULES, math_calls, _cuda_subscripts,
                     _cuda_split, _cuda_lhs, _cuda_floats),
    "tirx": LangSpec(TIRX_RULES, tirx_math_calls, _tirx_subscripts,
                     _tirx_split, _tirx_lhs, _tirx_floats),
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


_SHUFFLE = re.compile(r"(?:__shfl\w*|\w*shfl\w*)\s*\(")


def _calls(line: str, start_re: re.Pattern) -> list[str]:
    return _balanced_calls(line, start_re)


def _strip_subscripts(text: str) -> str:
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"\[[^\[\]]*\]", "[]", text)
    return text


def _pair_rules(spec: LangSpec, o_ln: int, old: str, new: str, floats: set[str] = frozenset()) -> list[Label]:
    ev = f"{old}  ->  {new}"
    labels = []
    for c, r, rx in spec.rules:
        if not (re.search(rx, old) or re.search(rx, new)):
            continue
        if r == "warp-shuffle" and _calls(old, _SHUFFLE) == _calls(new, _SHUFFLE):
            continue                       # the shuffle itself is unchanged: the edit is around it
        labels.append(Label(c, r, o_ln, ev))
    if spec.math_calls(old) != spec.math_calls(new):
        labels.append(Label("numerical", "math-call", o_ln, ev))
    rem, add = _token_delta(old, new)
    changed = set(rem) | set(add)
    if changed & _COMPARISON:
        labels.append(Label("indexing", "comparison-op", o_ln, ev))
    if "1" in changed and ("+" in changed or "-" in changed):
        labels.append(Label("indexing", "plus-minus-one", o_ln, ev))
    ho, so_ = spec.split(old)
    hn, sn_ = spec.split(new)
    for (ko, co, fo), (kn, cn, fn) in zip(ho, hn):
        if ko in ("for", "while") and kn in ("for", "while"):
            if co != cn:
                labels.append(Label("indexing", "loop-bound", o_ln, ev))
            elif fo != fn:
                labels.append(Label("indexing", "loop-step", o_ln, ev))
        elif ko == kn == "if" and co != cn:
            labels.append(Label("indexing", "guard", o_ln, ev))
    if len(ho) != len(hn):
        labels.append(Label("indexing", "guard" if "if" in {k for k, _, _ in ho + hn} else "loop-bound", o_ln, ev))
    if not labels and so_ != sn_ and _strip_subscripts(so_) != _strip_subscripts(sn_):   # fallback rule
        lhs = spec.lhs(sn_) if spec.lhs(sn_) is not None else spec.lhs(so_)
        if lhs is not None:
            is_float = lhs == "" or lhs in floats or (not floats and re.search(r"\bfloat\b|\d\.\d*f?\b", sn_))
            labels.append(Label("numerical", "arith", o_ln, ev) if is_float else Label("indexing", "index-arith", o_ln, ev))
    so, sn = spec.subscripts(old), spec.subscripts(new)
    if so and len(so) == len(sn) and so != sn:
        labels.append(Label("indexing", "index-expr", o_ln, ev))
    return labels


def classify(hunks: list[Hunk], lang: str = "cuda", src: str = "") -> list[Label]:
    spec = LANGS[lang]
    floats = spec.float_names(src) if src else set()
    labels: list[Label] = []
    for h in hunks:
        if h.kind == "mod":
            pairs, unpaired = _pair_lines(h.old, h.new)
            for (o_ln, o), (_, n) in pairs:
                labels.extend(_pair_rules(spec, o_ln, o, n, floats))
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
