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
  floor     ALWAYS added (not counted in the budget): sizes 1, T-1, T+1, 1000 (row ops) or 1x1x1, (T+1)x(T-1)xT,
            100x100x40 (tile2d), plus one extreme distribution, so a misclassified edit still meets a boundary;
            plus both sides of every size regime switch in the kernel (extract_thresholds): a kernel that takes
            another code path for large rows (`use_smem = N <= SMEM_FLOATS`) gets N = 8192 and 8193, because
            the fixed suite (N <= 4096) never reaches that path (GPU mutation run: 4 edits there went unseen)

VOLTA runs only CTA 0 at exact sizes, so volta_shapes = default + "deep" shape (exercises unrolled /
multi-iteration paths; GLM pilot: a 4x-unrolled loop that only runs for N > 384 hid a bug) + shapes that put a
tile boundary inside CTA 0. Shapes are capped by the manifest (layernorm VCs grow ~N^2).
"""
from __future__ import annotations

import ast
import operator
import re
import zlib

from ...task.testcase import TestCase
from ...task.workload import load_workload

_TILE_NAMES = r"(?:TILE\w*|BLOCK\w*|BM|BN|BK|THREADS\w*|NTHREADS|TPB)"
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


# ---- size regimes: `N <= SMEM_FLOATS`, `(n * 4 + NWARP * 4) <= (48 * 1024)` -> the N where the branch flips ----
REGIME_CAP = {"row": {"N": 1 << 16, "rows": 4096}, "tile2d": {"M": 4096, "N": 4096, "K": 4096}}
MAX_REGIMES = 4
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.LShift: operator.lshift, ast.RShift: operator.rshift}
_CMPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_SIZEOF = {"float": 4, "int": 4, "unsigned": 4, "double": 8, "half": 2, "__half": 2, "float2": 8, "float4": 16}
_CONST_DEFS = {
    "cuda": [r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+([^\n]+?)[ \t]*(?://[^\n]*)?$",
             r"\b(?:constexpr|const)\s+(?:unsigned\s+)?(?:int|size_t|long)\s+(\w+)\s*=\s*([^;]+);"],
    "tirx": [r"^[ \t]*([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*([^\n#]+?)[ \t]*(?:#[^\n]*)?$"],
}
_CMP_OP = re.compile(r"(?<![<>=!\-])(<=|>=|<|>)(?![<>=])")
_KEYWORD = re.compile(r"\b(?:and|or|not|if|else|elif|return|while|for|in|bool|int|const|auto|unsigned)\b")


def _c_to_py(expr: str) -> str:
    expr = re.sub(r"\bsizeof\s*\(\s*(\w+)\s*\)", lambda m: str(_SIZEOF.get(m.group(1), m.group(0))), expr)
    expr = re.sub(r"\(\s*(?:unsigned\s+)?(?:int|size_t|long|float)\s*\)", "", expr)          # C casts
    return re.sub(r"(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)[uUlLfF]+\b", r"\1", expr)               # 1024u, 4.0f


def _eval(node, env: dict):
    """Arithmetic and one comparison over integer/float constants and names in env; anything else raises."""
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= b <= 62:
            raise ValueError("shift")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPS:
        return _CMPS[type(node.ops[0])](_eval(node.left, env), _eval(node.comparators[0], env))
    raise ValueError(type(node).__name__)


def _constants(src: str, lang: str) -> dict:
    defs = [(m.group(1), m.group(2)) for pat in _CONST_DEFS.get(lang, []) for m in re.finditer(pat, src, re.M)]
    env: dict = {}
    for _ in range(3):                                   # a few passes: NWARP = BLOCK // 32 after BLOCK
        for name, expr in defs:
            if name in env:
                continue
            try:
                v = _eval(ast.parse(_c_to_py(expr).strip(), mode="eval"), env)
            except Exception:  # noqa: BLE001  (not a constant expression)
                continue
            if type(v) in (int, float):
                env[name] = v
    return env


def _operand(s: str, i: int, step: int) -> str:
    """The operand next to a comparison operator: scan from s[i] in direction step until an unmatched bracket,
    a statement / logic separator, or an assignment."""
    depth, j = 0, i
    opening, closing = ("(", ")") if step > 0 else (")", "(")
    while 0 <= j < len(s):
        ch = s[j]
        if ch == opening:
            depth += 1
        elif ch == closing:
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and ch in ";,?:{}&|!=":
            break
        j += step
    part = s[j + 1:i + 1] if step < 0 else s[i:j]
    pieces = _KEYWORD.split(part)
    return (pieces[-1] if step < 0 else pieces[0]).strip()


def extract_thresholds(src: str, lang: str, shape_keys: list[str], kind: str = "row") -> list[tuple[str, int]]:
    """[(shape key, v)]: the kernel compares that size with a constant and the comparison flips between v and v+1.
    Only comparisons whose other names all resolve to constants (#define, constexpr, UPPER_CASE = ...) count, so
    loop bounds like `i < N` are ignored. Evaluation is a small AST interpreter (no eval of kernel text)."""
    env = _constants(src, lang)
    caps = REGIME_CAP.get(kind, {})
    # a size is a name equal to a shape key, or a function parameter equal to one ignoring case (tirx's `n`);
    # a loop variable `k` is not the size K
    if lang == "cuda":
        params = set(re.findall(r"\b(?:const\s+)?(?:unsigned\s+)?(?:int|long|size_t)\s+(\w+)\s*[,)]", src))
    else:
        params = {p.split(":")[0].split("=")[0].strip() for m in re.finditer(r"\bdef\s+\w+\s*\(([^)]*)\)", src)
                  for p in m.group(1).split(",")}
    keys = {k: k for k in shape_keys} | {p: k for p in params for k in shape_keys if p.lower() == k.lower()}
    found: list[tuple[str, int]] = []
    comment = "//" if lang == "cuda" else "#"
    for line in src.splitlines():
        code = line.split(comment, 1)[0]
        if lang == "cuda" and code.lstrip().startswith("#"):
            continue
        for m in _CMP_OP.finditer(code):
            left, right = _operand(code, m.start() - 1, -1), _operand(code, m.end(), 1)
            if not left or not right:
                continue
            try:
                tree = ast.parse(_c_to_py(f"{left} {m.group(1)} {right}"), mode="eval")
            except SyntaxError:
                continue
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            free = [n for n in names if n not in env]
            if len(free) != 1 or free[0] not in keys:
                continue
            var, key = free[0], keys[free[0]]
            cap = caps.get(key, 4096)
            f = lambda v: bool(_eval(tree, {**env, var: v}))   # noqa: E731
            try:
                lo, hi = 1, cap
                if f(lo) == f(hi):
                    continue
                while hi - lo > 1:                     # monotone in the size: bisect the flip
                    mid = (lo + hi) // 2
                    lo, hi = (mid, hi) if f(mid) == f(lo) else (lo, mid)
            except Exception:  # noqa: BLE001  (division by zero etc.)
                continue
            if (key, lo) not in found:
                found.append((key, lo))
    return found[:MAX_REGIMES]


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
        return [tc(shape, dist=d, why=f"value range '{d}'") for d in wl.numerical_dists]
    raise ValueError(cat)


def floor_tests(wl, tiles: list[int], regimes: list[tuple[str, int]] | None = None) -> list[TestCase]:
    """Run on EVERY revision, whatever the classifier says: a few boundary sizes around the main tile, one extreme
    value distribution, and both sides of each size regime switch. A misclassified edit then still meets a
    non-divisible size, and an edit on a large-size code path still runs."""
    op, t = wl.op, (tiles or wl.tiles)[0]
    stress = wl.numerical_dists[0]
    tc = lambda shape, dist="normal", why="runs every revision": TestCase(  # noqa: E731
        op, shape, dist, _seed(op, "floor", sorted(shape.items()), dist), 1, f"floor: {shape} {dist} ({why})")
    row = wl.launch["kind"] == "row"
    if row:
        out = [tc({"rows": 2, "N": n}) for n in sorted({1, t - 1, t + 1, 1000}) if n >= 1]
        out.append(tc({"rows": 2, "N": 1000}, stress))
    else:
        out = [tc(s) for s in ({"M": 1, "N": 1, "K": 1}, {"M": t + 1, "N": t - 1, "K": t}, {"M": 100, "N": 100, "K": 40})]
        out.append(tc({"M": 100, "N": 100, "K": 40}, stress))
    for key, v in regimes or []:
        for n in (v, v + 1):
            shape = {**({"rows": 2, "N": 1000} if row else {"M": 33, "N": 33, "K": 33}), key: n}
            out.append(tc(shape, why=f"the kernel switches code path between {key}={v} and {key}={v + 1}"))
    return out


def _same_test(a: TestCase, b: TestCase) -> bool:
    return a.shape == b.shape and a.dist == b.dist and a.misalign == b.misalign


def generate_tests(workload: str, cats: list[str], tiles: list[int] | None = None, budget: int = 16,
                   regimes: list[tuple[str, int]] | None = None) -> list[TestCase]:
    """Targeted tests for the categories (round-robin, at most `budget`), then the floor tests not already present.
    regimes: extract_thresholds(new_src, ...) of the candidate."""
    wl = load_workload(workload)
    tiles = sorted(set(wl.tiles) | set(tiles or []))
    out, seen = [], set()
    per_cat = [_tests_for(wl, c, tiles) for c in cats]
    for i in range(max((len(p) for p in per_cat), default=0)):   # round-robin: every category gets tests first
        for p in per_cat:
            if i < len(p) and p[i].key() not in seen and len(out) < budget:
                seen.add(p[i].key())
                out.append(p[i])
    floor = []
    for f in floor_tests(wl, tiles, regimes):
        if not any(_same_test(f, t) for t in out + floor):
            floor.append(f)
    return out + floor


def volta_shapes(workload: str, cats: list[str], tiles: list[int] | None = None) -> list[dict]:
    wl = load_workload(workload)
    v = wl.volta
    if v is None:
        return []
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
