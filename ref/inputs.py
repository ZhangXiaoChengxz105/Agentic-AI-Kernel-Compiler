"""Input value distributions shared by the fixed tests (Task 1) and change-aware tests (Task 2).

make_tensor(shape, dist, gen) -> float32 CPU tensor, deterministic for a given torch.Generator.
Generated on CPU so a seed gives identical data on every machine; callers move it to the GPU.

dist          values                                  what it stresses
normal        N(0, 1)                                 typical data (fixed test suite)
uniform       U[-1, 1]                                typical data
large         N(0, 1) * 100                           exp overflow (softmax without max subtraction)
neg_large     -200 + N(0, 1)                          exp underflow to 0 -> 0/0 if not max-shifted
offset        1e4 + N(0, 1)                           catastrophic cancellation (one-pass variance)
tiny          N(0, 1) * 1e-39                         float32 subnormals
mixed         N(0, 1) * 10**U[-6, 6]                  very wide dynamic range in one row
constant      3.0 everywhere                          zero variance / ties in max
"""
from __future__ import annotations

import torch

DISTS = ("normal", "uniform", "large", "neg_large", "offset", "tiny", "mixed", "constant")


def make_tensor(shape: tuple[int, ...], dist: str, gen: torch.Generator) -> torch.Tensor:
    if dist == "normal":
        t = torch.randn(shape, generator=gen)
    elif dist == "uniform":
        t = torch.rand(shape, generator=gen) * 2 - 1
    elif dist == "large":
        t = torch.randn(shape, generator=gen) * 100
    elif dist == "neg_large":
        t = torch.randn(shape, generator=gen) - 200
    elif dist == "offset":
        t = torch.randn(shape, generator=gen, dtype=torch.float64) + 1e4
    elif dist == "tiny":
        t = torch.randn(shape, generator=gen, dtype=torch.float64) * 1e-39
    elif dist == "mixed":
        t = torch.randn(shape, generator=gen) * torch.pow(10.0, torch.rand(shape, generator=gen) * 12 - 6)
    elif dist == "constant":
        t = torch.full(shape, 3.0)
    else:
        raise ValueError(f"unknown dist {dist!r}; choose from {DISTS}")
    return t.to(torch.float32).contiguous()
