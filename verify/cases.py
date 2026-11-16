"""Building block 3: per-operator VOLTA setups (memory layout, params, launch dims).

softmax_case(ref, cand, N) describes ONE comparison of a row-wise softmax pair at row length N.

Memory model given to VOLTA (only CTA 0 = row 0 is executed):
  x : in,  float32, 2 rows * N   at 0x0010_0000   (row 1 exists so an off-by-one read past row 0
                                                  reads real neighbouring data, like on a GPU)
  y : out, float32, 2 rows * N   at 0x0800_0000   (only row 0 gets written by CTA 0)
  params: (x, y, N) in kernel declaration order.
Launch: reference uses its manifest block size (1), candidate its own (128); grid = (2,).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import WORKSPACE
from .volta import ArraySpec, VoltaCase

X_BASE = 0x0010_0000
Y_BASE = 0x0800_0000
ROWS = 2


@dataclass(frozen=True)
class Variant:
    name: str
    file: Path
    role: str
    block: int
    bug: dict | None
    expect_volta: str | None


def load_manifest(op_dir: str | Path) -> tuple[dict, list[Variant]]:
    op_dir = Path(op_dir)
    meta = json.loads((op_dir / "manifest.json").read_text())
    variants = [
        Variant(v["name"], op_dir / v["file"], v["role"], int(v["block"]), v.get("bug"), v.get("expect_volta"))
        for v in meta["variants"]
    ]
    return meta, variants


def softmax_case(ref_ptx: Path, ref: Variant, cand_ptx: Path, cand: Variant, N: int,
                 timeout_s: float = 600.0) -> VoltaCase:
    if N < 1:
        raise ValueError("N must be >= 1")
    if Y_BASE < X_BASE + 4 * ROWS * N:
        raise ValueError(f"N={N} too large for the fixed layout")
    return VoltaCase(
        label=f"{ref.name} vs {cand.name} @N={N}",
        ref_ptx=ref_ptx, ref_kernel=ref.name,
        cand_ptx=cand_ptx, cand_kernel=cand.name,
        ref_block=(ref.block,), cand_block=(cand.block,),
        grid=(ROWS,),
        arrays=(
            ArraySpec("x", X_BASE, 4, ROWS * N, "in"),
            ArraySpec("y", Y_BASE, 4, ROWS * N, "out"),
        ),
        params=("ptr:x", "ptr:y", f"int:{N}"),
        check_arrays=("y",),
        timeout_s=timeout_s,
    )


SOFTMAX_DIR = WORKSPACE / "kernels" / "softmax"
