"""Task 2.1: what changed between two kernel revisions?

diff_sources(old_src, new_src) -> list[Hunk]

Input : the two CUDA source texts (agent revision k and k+1, or a correct kernel and its bug-injected copy).
Output: Hunks, each a group of consecutive changed lines: kind "add" | "del" | "mod", with
        (line number, normalized text) for the old and the new side.
Normalization before diffing (so cosmetic edits do not count as changes):
  * // and /* */ comments removed, whitespace collapsed, blank lines dropped
  * the name of each `__global__` function replaced by KERNEL (renaming is not a semantic change)
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


def normalize(src: str) -> list[tuple[int, str]]:
    """Return [(original 1-based line number, normalized text)] for every non-empty code line."""
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


def diff_sources(old_src: str, new_src: str) -> list[Hunk]:
    a, b = normalize(old_src), normalize(new_src)
    sm = difflib.SequenceMatcher(a=[t for _, t in a], b=[t for _, t in b], autojunk=False)
    hunks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        kind = {"replace": "mod", "delete": "del", "insert": "add"}[tag]
        hunks.append(Hunk(kind, a[i1:i2], b[j1:j2]))
    return hunks
