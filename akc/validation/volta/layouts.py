"""Per-operator VOLTA setups (memory layout, params, launch dims) + expectations.

make_case(op, ref_ptx, ref, cand_ptx, cand, shape) -> VoltaCase for ONE comparison.

layout_for(op) -> OpLayout, as a function of the shape dict:
  arrays : (name, elem_bytes, length, kind) in kernel-parameter order; base addresses are assigned
           automatically (64 KiB-aligned, 64 KiB gap) so arrays can never overlap
  params : the kernel parameters in declaration order ("ptr:<array>", "int:<v>", "float:<v>")
  grid   : launch grid (only CTA 0 is executed by VOLTA; grid only matters if the kernel reads %nctaid)
  check  : output arrays to compare
The layout is DERIVED from the workload's OpSpec (make_inputs -> input arrays, out_shape -> output array,
kernel_args -> parameter order and scalar values, Workload.launch_dims -> grid); the output array is named after
the manifest signature's non-const pointer ("out" if there is none). OVERRIDES holds hand-written layouts for
operators whose kernel arguments cannot be derived that way (none so far).

Row-wise ops (shape key "rows") get 2 rows of input so an off-by-one read past row 0 reads real
neighbouring data (as on a GPU). Only row 0 (CTA 0) is written and compared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...task.workload import Variant
from .runner import ArraySpec, VoltaCase

BASE0 = 0x0010_0000
ALIGN = 0x1_0000
ROWS = 2                       # rows given to VOLTA for row-wise ops
FAIL_STATUSES = ("RACE", "DEADLOCK", "OUT_OF_BOUNDS", "UNINIT", "FOOTPRINT", "NOT_EQUIV", "DIV_BY_ZERO")


@dataclass(frozen=True)
class OpLayout:
    arrays: Callable[[dict], list[tuple[str, int, int, str]]]
    params: Callable[[dict], list[str]]
    grid: Callable[[dict], tuple[int, ...]]
    check: tuple[str, ...]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


OVERRIDES: dict[str, OpLayout] = {}


def _output_name(wl) -> str:
    sig = (wl.meta.get("signature") or {}).get("cuda", "")
    params = sig[sig.find("(") + 1:sig.rfind(")")] if "(" in sig else ""
    for p in params.split(","):
        m = re.match(r"\s*(const\s+)?\w+\s*\*\s*(?:__restrict__\s*)?(\w+)\s*$", p)
        if m and not m.group(1):
            return m.group(2)
    return "out"


def _volta_shape(op_spec, shape: dict) -> dict:
    s = dict(shape)
    if "rows" in op_spec.shape_keys:
        s["rows"] = ROWS
    return s


def derive_layout(op: str) -> OpLayout:
    """OpLayout from the OpSpec + manifest (see module docstring)."""
    import ctypes

    import torch

    from ...task.reference import OPS
    from ...task.workload import load_workload
    spec, wl = OPS[op], load_workload(op)
    out_name = _output_name(wl)

    def _args(shape: dict):
        s = _volta_shape(spec, shape)
        inputs = spec.make_inputs(s, "normal", 0)
        out = torch.empty(spec.out_shape(s), dtype=torch.float32)
        named = {t.data_ptr(): (n, t) for n, t in inputs.items()}
        named[out.data_ptr()] = (out_name, out)
        return s, named, spec.kernel_args(inputs, out, s)

    def arrays(shape: dict):
        _, named, args = _args(shape)
        out = []
        for v, ct in args:
            if ct is ctypes.c_void_p:
                n, t = named[v]
                out.append((n, t.element_size(), t.numel(), "out" if n == out_name else "in"))
        return out

    def params(shape: dict):
        _, named, args = _args(shape)
        kinds = {ctypes.c_void_p: "ptr", ctypes.c_int: "int", ctypes.c_float: "float"}
        return [f"ptr:{named[v][0]}" if ct is ctypes.c_void_p else f"{kinds[ct]}:{v}" for v, ct in args]

    def grid(shape: dict):
        g = wl.launch_dims(_volta_shape(spec, shape), (1,))[0]
        return tuple(g[:len(wl.launch["block"])])          # as many grid dims as the launch block has

    return OpLayout(arrays=arrays, params=params, grid=grid, check=(out_name,))


def layout_for(op: str) -> OpLayout:
    return OVERRIDES.get(op) or derive_layout(op)


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
    lay = layout_for(op)
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
