"""Task 2.3: turn change categories into targeted tests (run by a test executor) and VOLTA shapes.

extract_tiles(src, lang)                    tile/block sizes written in the candidate itself
generate_tests(workload, cats, tiles, budget) -> list[TestCase]
volta_shapes(workload, cats, tiles)         -> list[dict]

All operator knowledge comes from the workload manifest (launch kind, default tiles, numerical
distributions, VOLTA default/deep shapes and size cap), so a new workload needs no code change here.
Tiles = manifest tiles + sizes found in the candidate (e.g. the agent changed TILE to 32), so boundary shapes
follow the kernel's real tiling.

Strategy per category (B = row-op block / tile, T = 2-D tile):
  tiling    sizes around every tile: 1, T-1, T+1, 2T-1, 2T+1, and a non-power-of-two (1000)
            tile2d: K % T != 0, M/N not multiples of T, a matrix smaller than one tile
  indexing  same boundary sizes, with rows > 1 so off-by-one reads hit real neighbouring data
  memory    odd sizes (N % 4 != 0) and misaligned pointers (misalign=1 element: breaks float4 alignment)
  sync      large sizes and repeats (nondeterminism = race), plus a grid sweep (1 CTA with many
            iterations, many CTAs with few) -- GPU tests are weak at races; VOLTA is the main race check
  numerical value distributions from the manifest at a modest shape
  (none)    a small sanity set (one non-divisible shape + one stress distribution)

VOLTA runs only CTA 0 at exact sizes, so volta_shapes = default + "deep" shape (exercises unrolled /
multi-iteration paths; GLM pilot: a 4x-unrolled loop that only runs for N > 384 hid a bug) + shapes that put a
tile boundary inside CTA 0. Shapes are capped by the manifest (layernorm VCs grow ~N^2).
"""
from __future__ import annotations

import re
import zlib

from ...task.testcase import TestCase
from ...task.workload import load_workload

_TILE_NAMES = r"(?:TILE\w*|BLOCK\w*|BM|BN|BK|THREADS\w*|NTHREADS)"
_TILE_PATTERNS = {
    "cuda": [rf"#\s*define\s+{_TILE_NAMES}\s+(\d+)", rf"(?:constexpr|const)\s+(?:int|unsigned)\s+{_TILE_NAMES}\s*=\s*(\d+)"],
    "tirx": [rf"^\s*{_TILE_NAMES}\s*=\s*(\d+)\s*$", r"LaunchConfig\([^)]*block\s*=\s*(\d+)"],
}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def extract_tiles(src: str, lang: str = "cuda") -> list[int]:
    found = set()
    for pat in _TILE_PATTERNS.get(lang, []):
        for m in re.finditer(pat, src, re.M):
            v = int(m.group(1))
            if 8 <= v <= 4096:
                found.add(v)
    return sorted(found)


def _boundary(tiles: list[int]) -> list[int]:
    sizes = {1, 1000}
    for t in tiles:
        sizes |= {t - 1, t + 1, 2 * t - 1, 2 * t + 1}
    return sorted(s for s in sizes if s >= 1)


def _tests_for(wl, cat: str, tiles: list[int]) -> list[TestCase]:
    op, kind = wl.op, wl.launch["kind"]
    tc = lambda shape, dist="normal", repeat=1, why="", mis=0: TestCase(  # noqa: E731
        op, shape, dist, _seed(op, cat, sorted(shape.items()), dist, mis), repeat, f"{cat}: {why}", mis)
    if cat in ("tiling", "indexing"):
        if kind == "row":
            rows = 3 if cat == "indexing" else 2
            return [tc({"rows": rows, "N": n}, why=f"N={n} vs tiles {tiles}") for n in _boundary(tiles)]
        out = []
        for t in tiles:
            out += [tc(s, why=f"{s} vs tile {t}") for s in (
                {"M": t - 4, "N": t - 4, "K": 2 * t + 8}, {"M": 6 * t + 4, "N": 6 * t + 4, "K": 2 * t + 8},
                {"M": 2 * t + 1, "N": 3 * t - 1, "K": 4 * t - 3}, {"M": t + 1, "N": t - 1, "K": t})]
        return out + [tc({"M": 1, "N": 1, "K": 1}, why="1x1x1")]
    if cat == "memory":
        if kind == "row":
            return ([tc({"rows": 3, "N": n}, why=f"N={n} (N % 4 = {n % 4})") for n in (127, 1001, 1023)] +
                    [tc({"rows": 3, "N": n}, why=f"N={n}, pointers misaligned by 1 element", mis=1) for n in (1024, 1000)])
        return [tc({"M": 33, "N": 63, "K": 31}, why="odd sizes"),
                tc({"M": 64, "N": 64, "K": 64}, why="pointers misaligned by 1 element", mis=1)]
    if cat == "sync":
        if kind == "row":
            return [tc({"rows": 256, "N": 4096}, repeat=5, why="many iterations, repeated x5"),
                    tc({"rows": 256, "N": 4099}, repeat=5, why="ragged, repeated x5"),
                    tc({"rows": 1, "N": 8192}, repeat=10, why="grid sweep: 1 CTA, many iterations, x10"),
                    tc({"rows": 2048, "N": 129}, repeat=5, why="grid sweep: many CTAs, 2 iterations, x5")]
        return [tc({"M": 512, "N": 512, "K": 512}, repeat=5, why="many K-tiles, repeated x5"),
                tc({"M": 257, "N": 263, "K": 1031}, repeat=5, why="ragged + many K-tiles, repeated x5"),
                tc({"M": 16, "N": 16, "K": 2048}, repeat=10, why="grid sweep: 1 CTA, 128 K-tiles, x10")]
    if cat == "numerical":
        shape = {"rows": 8, "N": 1000} if kind == "row" else {"M": 128, "N": 128, "K": 256}
        return [tc(shape, dist=d, why=f"value range '{d}'") for d in wl.meta["numerical_dists"]]
    raise ValueError(cat)


def generate_tests(workload: str, cats: list[str], tiles: list[int] | None = None, budget: int = 16) -> list[TestCase]:
    wl = load_workload(workload)
    tiles = sorted(set(wl.tiles) | set(tiles or []))
    if not cats:
        sanity = {"rows": 2, "N": 1000} if wl.launch["kind"] == "row" else {"M": 100, "N": 100, "K": 40}
        stress = wl.meta["numerical_dists"][0]
        return [TestCase(workload, sanity, "normal", _seed(workload, "none"), 1, "no change category: non-divisible sanity"),
                TestCase(workload, sanity, stress, _seed(workload, "none", stress), 1, f"no change category: '{stress}' values")]
    per_cat = [_tests_for(wl, c, tiles) for c in cats]
    out, seen = [], set()
    for i in range(max(len(p) for p in per_cat)):          # round-robin: every category gets tests first
        for p in per_cat:
            if i < len(p) and p[i].key() not in seen and len(out) < budget:
                seen.add(p[i].key())
                out.append(p[i])
    return out


def volta_shapes(workload: str, cats: list[str], tiles: list[int] | None = None) -> list[dict]:
    wl = load_workload(workload)
    v = wl.meta["volta"]
    tiles = sorted(set(wl.tiles) | set(tiles or []))
    shapes = [dict(v["default"])] + ([dict(v["deep"])] if v.get("deep") else [])
    shape_cats = {"tiling", "indexing", "memory"} & set(cats)
    if wl.launch["kind"] == "row":
        for b in tiles[:2]:
            if shape_cats:
                shapes += [{"N": b + b // 2 + 8}] + ([{"N": b - 28}] if b > 32 else [])
            if "sync" in cats:
                shapes.append({"N": 2 * b + 8})
    else:
        for t in tiles[:2]:
            if shape_cats:
                shapes += [{"M": 2 * t, "N": 2 * t, "K": 2 * t + 8}, {"M": t - 4, "N": t - 4, "K": 2 * t + 8}]
            if "sync" in cats:
                shapes.append({"M": t, "N": t, "K": 4 * t})
    return wl.cap_volta_shapes(shapes)
