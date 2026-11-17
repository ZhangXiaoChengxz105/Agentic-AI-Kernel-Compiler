"""Task 2.1: what changed between two kernel revisions?

diff_sources(old_src, new_src, lang=None) -> list[Hunk]      lang: "cuda" | "tirx" (auto-detected)

Input : the two kernel source texts (CUDA C++ or tirx-lite Python) (agent revision k and k+1, or a correct kernel and its bug-injected copy).
Output: Hunks, each a group of consecutive changed lines: kind "add" | "del" | "mod", with
        (line number, normalized text) for the old and the new side.
Normalization before diffing (so cosmetic edits do not count as changes). CUDA:
  * // and /* */ comments removed, whitespace collapsed, blank lines dropped
  * the name of each `__global__` function replaced by KERNEL (renaming is not a semantic change)
tirx-lite: the Python parser does it (normalize_python): one entry per statement, headers of compound
statements, comments/docstrings/formatting gone, @txl.kernel functions renamed KERNEL.
Cosmetic revisions give NO hunks: if the two token streams are equal up to line breaks and a consistent
(one-to-one) renaming of local identifiers, nothing changed semantically. Keywords, CUDA builtins, math
functions, `__`-names and attribute names (after a dot) may not be renamed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//.*?$", re.M)
_KERNEL_NAME = re.compile(r"(__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?)([A-Za-z_]\w*)")


@dataclass(frozen=True)
class Hunk:
    kind: str                                     # add | del | mod
    old: list = field(default_factory=list)       # [(lineno, text)]
    new: list = field(default_factory=list)

    def old_text(self) -> str:
        return "\n".join(t for _, t in self.old)

    def new_text(self) -> str:
        return "\n".join(t for _, t in self.new)


def detect_lang(src: str) -> str:
    """'tirx' for tirx-lite Python modules, else 'cuda'."""
    return "tirx" if re.search(r"^\s*(import|from)\s+tirx_kernels", src, re.M) else "cuda"


def normalize_python(src: str) -> list[tuple[int, str]]:
    """tirx-lite (Python) normalization via the Python parser: one entry per statement, compound
    statements contribute only their header (`with txl.if_(col < n), txl.then_():`), comments,
    docstrings and formatting disappear, and functions decorated with txl.kernel are renamed KERNEL."""
    import ast
    import copy
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.stmt):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue                                                     # docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                "kernel" in ast.unparse(d) for d in node.decorator_list):
            node = copy.copy(node)
            node.name = "KERNEL"
        if isinstance(getattr(node, "body", None), list):
            head = copy.copy(node)
            head.body = [ast.Pass()]
            for attr in ("orelse", "finalbody", "handlers"):
                if hasattr(head, attr):
                    setattr(head, attr, [])
            if hasattr(head, "decorator_list"):
                head.decorator_list = []
            text = ast.unparse(head).splitlines()[0]
        else:
            text = ast.unparse(node)
        out.append((node.lineno, " ".join(text.split())))
    return sorted(out)


def normalize(src: str, lang: str = "cuda") -> list[tuple[int, str]]:
    """Return [(original 1-based line number, normalized text)] for every non-empty code line."""
    if lang == "tirx":
        return normalize_python(src)
    # blank out comments but keep newlines so line numbers survive
    src = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src)
    src = _LINE_COMMENT.sub("", src)
    src = _KERNEL_NAME.sub(lambda m: m.group(1) + "KERNEL", src)
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        t = " ".join(line.split())
        if t:
            out.append((i, t))
    return out


_TOK = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*(?:[eE][-+]?\d+)?[fFuUlL]?|\S")
_FIXED_NAMES = set("""
for while do if else switch case default break continue return goto int unsigned signed long short char bool
float double void const volatile static extern inline struct union typedef sizeof size_t auto true false
nullptr template typename define include ifndef ifdef endif undef pragma unroll threadIdx blockIdx blockDim
gridDim warpSize KERNEL INFINITY NAN FLT_MAX FLT_MIN EXP float2 float4 int2 int4 half bfloat16
import from as def with in range lambda None True False and or not is pass yield txl T tirx_kernels
exp expf exp2 exp2f log logf log2 log2f sqrt sqrtf rsqrt rsqrtf fmax fmaxf fmin fminf fma fmaf tanh tanhf pow powf
fabs fabsf max min abs
""".split())


def cosmetic_only(a: list[tuple[int, str]], b: list[tuple[int, str]]) -> bool:
    """True if the normalized sources differ only in line breaks / a consistent renaming of local names."""
    ta, tb = _TOK.findall(" ".join(t for _, t in a)), _TOK.findall(" ".join(t for _, t in b))
    if len(ta) != len(tb):
        return False
    fwd, back = {}, {}
    for i, (x, y) in enumerate(zip(ta, tb)):
        ident = re.fullmatch(r"[A-Za-z_]\w*", x) and re.fullmatch(r"[A-Za-z_]\w*", y)
        if x == y and not ident:
            continue
        fixed = x in _FIXED_NAMES or y in _FIXED_NAMES or x.startswith("__") or y.startswith("__") or \
            (i > 0 and ta[i - 1] == ".")
        if not ident or (fixed and x != y):
            return False
        if fwd.setdefault(x, y) != y or back.setdefault(y, x) != x:
            return False
    return True


def diff_sources(old_src: str, new_src: str, lang: str | None = None) -> list[Hunk]:
    lang = lang or detect_lang(old_src + "\n" + new_src)
    a, b = normalize(old_src, lang), normalize(new_src, lang)
    if cosmetic_only(a, b):
        return []
    sm = difflib.SequenceMatcher(a=[t for _, t in a], b=[t for _, t in b], autojunk=False)
    hunks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        kind = {"replace": "mod", "delete": "del", "insert": "add"}[tag]
        hunks.append(Hunk(kind, a[i1:i2], b[j1:j2]))
    return hunks
