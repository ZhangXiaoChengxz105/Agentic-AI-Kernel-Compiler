"""Building block 3: per-operator VOLTA setups (memory layout, params, launch dims) + expectations.

make_case(op, ref_ptx, ref, cand_ptx, cand, shape) -> VoltaCase for ONE comparison.

Each op declares, as a function of its shape dict:
  arrays : (name, elem_bytes, length, kind) in kernel-parameter order; base addresses are assigned
           automatically (64 KiB-aligned, 64 KiB gap) so arrays can never overlap
  params : the kernel parameters in declaration order ("ptr:<array>", "int:<v>", "float:<v>")
  grid   : launch grid (only CTA 0 is executed by VOLTA; grid only matters if the kernel reads %nctaid)
  check  : output arrays to compare

Row-wise ops (softmax, layernorm) get 2 rows of input so an off-by-one read past row 0 reads real
neighbouring data (as on a GPU). Only row 0 (CTA 0) is written and compared.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import WORKSPACE
from .volta import ArraySpec, VoltaCase

KERNELS_DIR = WORKSPACE / "kernels"
BASE0 = 0x0010_0000
ALIGN = 0x1_0000
ROWS = 2                       # rows given to VOLTA for row-wise ops
LN_EPS = "1e-05"
FAIL_STATUSES = ("RACE", "DEADLOCK", "OUT_OF_BOUNDS", "UNINIT", "FOOTPRINT", "NOT_EQUIV")


@dataclass(frozen=True)
class Variant:
    name: str
    file: Path
    role: str
    block: tuple[int, ...]
    bug: dict | None
    expect_volta: object          # str | list[str] | list[{"if": expr, "then": str|list}] | None


@dataclass(frozen=True)
class OpLayout:
    arrays: Callable[[dict], list[tuple[str, int, int, str]]]
    params: Callable[[dict], list[str]]
    grid: Callable[[dict], tuple[int, ...]]
    check: tuple[str, ...]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


LAYOUTS: dict[str, OpLayout] = {
    "softmax": OpLayout(
        arrays=lambda s: [("x", 4, ROWS * s["N"], "in"), ("y", 4, ROWS * s["N"], "out")],
        params=lambda s: ["ptr:x", "ptr:y", f"int:{s['N']}"],
        grid=lambda s: (ROWS,),
        check=("y",),
    ),
    "layernorm": OpLayout(
        arrays=lambda s: [("x", 4, ROWS * s["N"], "in"), ("w", 4, s["N"], "in"), ("b", 4, s["N"], "in"),
                          ("y", 4, ROWS * s["N"], "out")],
        params=lambda s: ["ptr:x", "ptr:w", "ptr:b", "ptr:y", f"int:{s['N']}", f"float:{s.get('eps', LN_EPS)}"],
        grid=lambda s: (ROWS,),
        check=("y",),
    ),
    "gemm": OpLayout(
        arrays=lambda s: [("A", 4, s["M"] * s["K"], "in"), ("B", 4, s["K"] * s["N"], "in"),
                          ("C", 4, s["M"] * s["N"], "out")],
        params=lambda s: ["ptr:A", "ptr:B", "ptr:C", f"int:{s['M']}", f"int:{s['N']}", f"int:{s['K']}"],
        grid=lambda s: (_cdiv(s["N"], 16), _cdiv(s["M"], 16)),
        check=("C",),
    ),
}


def op_dir(op: str) -> Path:
    return KERNELS_DIR / op


def load_manifest(op_or_dir: str | Path) -> tuple[dict, list[Variant]]:
    d = Path(op_or_dir) if Path(op_or_dir).is_dir() else op_dir(str(op_or_dir))
    meta = json.loads((d / "manifest.json").read_text())
    variants = []
    for v in meta["variants"]:
        block = v["block"] if isinstance(v["block"], list) else [v["block"]]
        variants.append(Variant(v["name"], d / v["file"], v["role"], tuple(int(b) for b in block),
                                v.get("bug"), v.get("expect_volta")))
    return meta, variants


def place_arrays(specs: list[tuple[str, int, int, str]]) -> tuple[ArraySpec, ...]:
    out, base = [], BASE0
    for name, width, length, kind in specs:
        if length < 1:
            raise ValueError(f"array {name} has length {length}")
        a = ArraySpec(name, base, width, length, kind)
        out.append(a)
        base = _cdiv(a.end + ALIGN, ALIGN) * ALIGN
    return tuple(out)


def shape_label(shape: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in shape.items())


def make_case(op: str, ref_ptx: Path, ref: Variant, cand_ptx: Path, cand: Variant, shape: dict,
              timeout_s: float = 600.0) -> VoltaCase:
    lay = LAYOUTS[op]
    return VoltaCase(
        label=f"{ref.name} vs {cand.name} @{shape_label(shape)}",
        ref_ptx=ref_ptx, ref_kernel=ref.name, cand_ptx=cand_ptx, cand_kernel=cand.name,
        ref_block=ref.block, cand_block=cand.block,
        grid=lay.grid(shape),
        arrays=place_arrays(lay.arrays(shape)),
        params=tuple(lay.params(shape)),
        check_arrays=lay.check,
        timeout_s=timeout_s,
    )


def expected_statuses(expect, shape: dict) -> tuple[str, ...] | None:
    """Resolve a manifest `expect_volta` entry for a concrete shape.

    Formats: "EQUIV" | ["RACE", "NOT_EQUIV"] (any of) |
             [{"if": "K % 16 == 0", "then": "EQUIV"}, {"then": "NOT_EQUIV"}] (first match wins).
    `if` expressions may only use the shape's variables (evaluated without builtins).
    """
    if expect is None:
        return None
    if isinstance(expect, str):
        return (expect,)
    if expect and all(isinstance(e, str) for e in expect):
        return tuple(expect)
    for rule in expect:
        cond = rule.get("if")
        if cond is None or eval(cond, {"__builtins__": {}}, dict(shape)):  # noqa: S307 (own manifest only)
            then = rule["then"]
            return (then,) if isinstance(then, str) else tuple(then)
    return None
