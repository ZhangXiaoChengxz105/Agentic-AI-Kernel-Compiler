"""PyTorch reference ("golden") implementations + everything the test executors need per workload.

OpSpec fields
  shape keys   : softmax/layernorm {"rows", "N"}; gemm {"M", "N", "K"}  (any positive sizes, incl.
                 non-divisible ones like N=1000 or K=40)
  make_inputs  : (shape, dist, seed) -> dict of float32 CPU tensors, deterministic
  golden       : inputs -> float64 result computed with PyTorch (the "standard answer")
  out_shape    : shape -> output tensor shape
  kernel_args  : (device inputs, device output, shape) -> [(value, ctype)] in kernel-signature order
  launch       : (shape, block from manifest) -> (grid, block)
  flops/bytes  : for the benchmark (1.6)
  tolerance    : float32 (atol, rtol); `magnitude` (when set) replaces |golden| as the quantity rtol multiplies,
                 so the bound follows the error any float32 implementation must make:
                 gemm      |A|@|B|: grows with K the way float accumulation error does;
                 layernorm |y| + (log2 N + 4) u |mean|/std |w| / rtol: rounding the row mean to float32
                           (u = 2^-24) is amplified by |mean|/std. Without it, rows with a large offset
                           (mean 1e4, std 1) fail even PyTorch's own float32 layer_norm (checked on the CPU:
                           ~2000 of 8000 outputs), while the one-pass-variance bug is still caught (NaN / O(1)).
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .inputs import make_tensor

LN_EPS = 1e-5
LN_RTOL = 1e-3
U32 = 2.0 ** -24          # float32 unit roundoff


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@dataclass(frozen=True)
class OpSpec:
    name: str
    shape_keys: tuple[str, ...]
    make_inputs: Callable[[dict, str, int], dict]
    golden: Callable[[dict], torch.Tensor]
    out_shape: Callable[[dict], tuple[int, ...]]
    kernel_args: Callable[[dict, torch.Tensor, dict], list]
    launch: Callable[[dict, tuple[int, ...]], tuple[tuple[int, int, int], tuple[int, int, int]]]
    flops: Callable[[dict], float]
    bytes: Callable[[dict], float]
    atol: float
    rtol: float
    torch_impl: Callable[[dict], torch.Tensor]       # fast PyTorch version (benchmark baseline)
    magnitude: Callable[[dict], torch.Tensor] | None = None


def _ptr(t: torch.Tensor):
    return (t.data_ptr(), ctypes.c_void_p)


def _dims3(t: tuple[int, ...]) -> tuple[int, int, int]:
    t = tuple(t) + (1,) * (3 - len(t))
    return t[0], t[1], t[2]


# ---- softmax ---------------------------------------------------------------------------------
def _softmax_inputs(s, dist, seed):
    return {"x": make_tensor((s["rows"], s["N"]), dist, _gen(seed))}


SOFTMAX = OpSpec(
    name="softmax", shape_keys=("rows", "N"),
    make_inputs=_softmax_inputs,
    golden=lambda i: torch.softmax(i["x"].double(), dim=-1),
    out_shape=lambda s: (s["rows"], s["N"]),
    kernel_args=lambda d, out, s: [_ptr(d["x"]), _ptr(out), (s["N"], ctypes.c_int)],
    launch=lambda s, block: ((s["rows"], 1, 1), _dims3(block)),
    flops=lambda s: 5.0 * s["rows"] * s["N"],
    bytes=lambda s: 8.0 * s["rows"] * s["N"],
    atol=1e-6, rtol=1e-4,
    torch_impl=lambda d: torch.softmax(d["x"], dim=-1),
)


# ---- layernorm -------------------------------------------------------------------------------
def _ln_inputs(s, dist, seed):
    g = _gen(seed)
    x = make_tensor((s["rows"], s["N"]), dist, g)
    w = (1.0 + 0.1 * torch.randn(s["N"], generator=g)).float()
    b = (0.1 * torch.randn(s["N"], generator=g)).float()
    return {"x": x, "w": w, "b": b}


def _ln_magnitude(i):
    x = i["x"].double()
    n = x.shape[-1]
    mean = x.mean(-1, keepdim=True)
    std = torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + LN_EPS)
    y = F.layer_norm(x, (n,), i["w"].double(), i["b"].double(), LN_EPS)
    unavoidable = (math.log2(max(n, 2)) + 4) * U32 * mean.abs() / std * i["w"].double().abs()
    return y.abs() + unavoidable / LN_RTOL


LAYERNORM = OpSpec(
    name="layernorm", shape_keys=("rows", "N"),
    make_inputs=_ln_inputs,
    golden=lambda i: F.layer_norm(i["x"].double(), (i["x"].shape[-1],), i["w"].double(), i["b"].double(), LN_EPS),
    out_shape=lambda s: (s["rows"], s["N"]),
    kernel_args=lambda d, out, s: [_ptr(d["x"]), _ptr(d["w"]), _ptr(d["b"]), _ptr(out),
                                   (s["N"], ctypes.c_int), (LN_EPS, ctypes.c_float)],
    launch=lambda s, block: ((s["rows"], 1, 1), _dims3(block)),
    flops=lambda s: 8.0 * s["rows"] * s["N"],
    bytes=lambda s: 4.0 * (2 * s["rows"] * s["N"] + 2 * s["N"]),
    atol=1e-4, rtol=LN_RTOL,
    torch_impl=lambda d: F.layer_norm(d["x"], (d["x"].shape[-1],), d["w"], d["b"], LN_EPS),
    magnitude=_ln_magnitude,
)


# ---- gemm ------------------------------------------------------------------------------------
def _gemm_inputs(s, dist, seed):
    g = _gen(seed)
    return {"A": make_tensor((s["M"], s["K"]), dist, g), "B": make_tensor((s["K"], s["N"]), dist, g)}


GEMM = OpSpec(
    name="gemm", shape_keys=("M", "N", "K"),
    make_inputs=_gemm_inputs,
    golden=lambda i: i["A"].double() @ i["B"].double(),
    out_shape=lambda s: (s["M"], s["N"]),
    kernel_args=lambda d, out, s: [_ptr(d["A"]), _ptr(d["B"]), _ptr(out),
                                   (s["M"], ctypes.c_int), (s["N"], ctypes.c_int), (s["K"], ctypes.c_int)],
    launch=lambda s, block: ((_cdiv(s["N"], 16), _cdiv(s["M"], 16), 1), _dims3(block)),
    flops=lambda s: 2.0 * s["M"] * s["N"] * s["K"],
    bytes=lambda s: 4.0 * (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]),
    atol=1e-5, rtol=1e-5,
    torch_impl=lambda d: d["A"] @ d["B"],
    magnitude=lambda i: i["A"].double().abs() @ i["B"].double().abs(),
)

OPS: dict[str, OpSpec] = {o.name: o for o in (SOFTMAX, LAYERNORM, GEMM)}
