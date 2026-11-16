"""Task 2.3: turn change categories into targeted tests (GPU harness) and VOLTA shapes.

generate_tests(op, cats, block)  -> list[TestCase]   (run with harness.runner)
volta_shapes(op, cats, block)    -> list[dict]       (run with verify.cases.make_case)

Inputs : op name, categories from classify.categories(), and the kernel's launch block from the
         manifest (the tile/block size the boundaries are built around: B = 128 for row ops,
         T = 16 for gemm).
Outputs: deterministic TestCases tagged with the reason they exist, capped at `budget`.

Strategy per category (B = block size along the reduced dim, T = gemm tile):
  tiling    sizes around tile boundaries: 1, B-1, B+1, 2B-1, 2B+1, a non-power-of-two (1000)
            gemm: K % T != 0, M/N not multiples of T, matrices smaller than one tile
  indexing  same boundary sizes + rows > 1 so off-by-one reads hit real neighbouring data
  memory    odd/unaligned sizes (N % 4 != 0) that break vector-width assumptions
  sync      large sizes (many loop iterations, many CTAs), repeat=5 (nondeterminism = race)
  numerical value distributions at a modest shape: softmax large/neg_large/mixed/constant,
            layernorm offset/constant/tiny/large, gemm large/mixed/tiny
  (none)    a small sanity set (one non-divisible shape + one stress distribution)

VOLTA only runs CTA 0 at the exact sizes it is given, so volta_shapes adds a "deep" shape (DEEP_VOLTA, always,
for softmax/gemm) that exercises unrolled/multi-iteration paths, and puts the boundary INSIDE CTA 0:
  row ops   N = B + B/2 + 8 (not a multiple of B, > 1 tile) and N = B - 28 (< 1 tile)
  gemm      K % T != 0 (K = 2T + 8) and M = N = T - 4 (< 1 tile, so CTA 0 is also the edge CTA)
VOLTA cannot see numerical problems (real-number semantics): for "numerical" it adds nothing beyond
the default shape; those bugs are left to the GPU tests.
"""
from __future__ import annotations

import zlib

from harness.testcase import TestCase

ROW_OPS = ("softmax", "layernorm")
NUMERICAL_DISTS = {
    "softmax": ("large", "neg_large", "mixed", "constant"),
    "layernorm": ("offset", "constant", "tiny", "large"),
    "gemm": ("large", "mixed", "tiny"),
}
DEFAULT_VOLTA = {"softmax": {"N": 256}, "layernorm": {"N": 128}, "gemm": {"M": 32, "N": 32, "K": 48}}
# "Deep" shape: large enough that unrolled / multi-iteration code paths run (GLM pilot: a kernel whose 4x-unrolled
# loop only runs for N > 384 dropped a term; VOLTA said EQUIV at N <= 384 and NOT_EQUIV at N = 385).
# Always added, because such a bug can come from an earlier revision than the one being diffed.
# Not for layernorm: its VCs grow ~N^2 (N >= 512 took > 3.5 min and ~10 GB RAM).
DEEP_VOLTA = {"softmax": {"N": 584}, "gemm": {"M": 32, "N": 32, "K": 72}}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def _row_sizes(B: int) -> list[int]:
    return [1, B - 1, B + 1, 2 * B - 1, 2 * B + 1, 1000]


def _gemm_shapes(T: int) -> list[dict]:
    return [{"M": T - 4, "N": T - 4, "K": 2 * T + 8}, {"M": 100, "N": 100, "K": 40},
            {"M": 33, "N": 47, "K": 61}, {"M": 1, "N": 1, "K": 1}, {"M": T + 1, "N": T - 1, "K": T}]


def _tests_for(op: str, cat: str, B: int, T: int) -> list[TestCase]:
    tc = lambda shape, dist="normal", repeat=1, why="": TestCase(  # noqa: E731
        op, shape, dist, _seed(op, cat, sorted(shape.items()), dist), repeat, f"{cat}: {why}")
    if cat in ("tiling", "indexing"):
        if op in ROW_OPS:
            rows = 3 if cat == "indexing" else 2
            return [tc({"rows": rows, "N": n}, why=f"N={n} vs block {B}") for n in _row_sizes(B)]
        return [tc(s, why=f"{s} vs tile {T}") for s in _gemm_shapes(T)]
    if cat == "memory":
        if op in ROW_OPS:
            return [tc({"rows": 3, "N": n}, why=f"N={n} (N % 4 = {n % 4})") for n in (127, 1001, 1023)]
        return [tc({"M": 33, "N": 63, "K": 31}, why="odd sizes")]
    if cat == "sync":
        if op in ROW_OPS:
            return [tc({"rows": 256, "N": n}, repeat=5, why=f"N={n}, many iterations, repeated x5")
                    for n in (4096, 4099)]
        return [tc({"M": 512, "N": 512, "K": 512}, repeat=5, why="many K-tiles, repeated x5"),
                tc({"M": 257, "N": 263, "K": 1031}, repeat=5, why="ragged + many K-tiles, repeated x5")]
    if cat == "numerical":
        shape = {"rows": 8, "N": 1000} if op in ROW_OPS else {"M": 128, "N": 128, "K": 256}
        return [tc(shape, dist=d, why=f"value range '{d}'") for d in NUMERICAL_DISTS[op]]
    raise ValueError(cat)


def generate_tests(op: str, cats: list[str], block: tuple[int, ...], budget: int = 16) -> list[TestCase]:
    B = block[0] if op in ROW_OPS else 128
    T = 16 if op == "gemm" else B
    if not cats:
        sanity = ({"rows": 2, "N": 1000} if op in ROW_OPS else {"M": 100, "N": 100, "K": 40})
        stress = NUMERICAL_DISTS[op][0]
        return [TestCase(op, sanity, "normal", _seed(op, "none"), 1, "no change category: non-divisible sanity"),
                TestCase(op, sanity, stress, _seed(op, "none", stress), 1, f"no change category: '{stress}' values")]
    per_cat = [_tests_for(op, c, B, T) for c in cats]
    out, seen = [], set()
    # round-robin across categories so every detected category gets tests before the budget runs out
    for i in range(max(len(p) for p in per_cat)):
        for p in per_cat:
            if i < len(p) and p[i].key() not in seen and len(out) < budget:
                seen.add(p[i].key())
                out.append(p[i])
    return out


def volta_shapes(op: str, cats: list[str], block: tuple[int, ...]) -> list[dict]:
    shapes = [dict(DEFAULT_VOLTA[op])]
    if op in DEEP_VOLTA:
        shapes.append(dict(DEEP_VOLTA[op]))
    if op in ROW_OPS:
        B = block[0]
        if {"tiling", "indexing", "memory"} & set(cats):
            shapes += [{"N": B + B // 2 + 8}, {"N": B - 28}]
        if "sync" in cats:
            shapes += [{"N": 2 * B + 8}]
    else:
        T = 16
        if {"tiling", "indexing", "memory"} & set(cats):
            shapes += [{"M": 2 * T, "N": 2 * T, "K": 2 * T + 8}, {"M": T - 4, "N": T - 4, "K": 2 * T + 8}]
        if "sync" in cats:
            shapes += [{"M": T, "N": T, "K": 4 * T}]
    uniq = []
    for s in shapes:
        if s not in uniq:
            uniq.append(s)
    return uniq
