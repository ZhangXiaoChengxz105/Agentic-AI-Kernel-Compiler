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
outlier[@P]   per row: N(0, 1) * row scale, plus ONE  reductions that skip or double-count positions, and
              outlier at column (r mod P) + P*k       reads from the neighbouring row (see _outlier)

The outlier family (operator-agnostic; a "row" is the last dimension, all leading dimensions are rows):
  * row scale alternates between adjacent rows: 1 and 1e3. A kernel that reads one element of the next or the
    previous row (off-by-one, wrong row stride) mixes in values 1e3 times larger or smaller than its own, so the
    wrong read changes the result far beyond tolerance (with N(0, 1) data the neighbour looks like the row itself).
  * exactly one outlier per row, at column (r mod P) + P*k with k seeded-random among the columns that exist
    (any seeded-random column if N < P). P is the kernel's own block / tile size (generate.py writes outlier@T for
    the first tile T), so with rows >= P every position modulo P -- every thread, lane or tile slot that a
    reduction might skip -- holds the outlier in some row. Plain "outlier" means P = N (column r mod N).
  * magnitude: a fixed ladder relative to the row scale, ladder[(r // 2) % 4] with ladder = +1e2, +1e3, -1e3, +1e4
    (rows r and r+1 share a rung, so with 8+ rows every rung meets both row scales). Chosen for being far above
    the noise, not for one operator:
      - lowest rung 1e2 x scale: the largest of up to 16384 N(0, 1) samples is about 4.5, so even the smallest
        outlier is >20x every other value in its row; including or missing it moves any max, sum or mean of the
        row by far more than any test tolerance;
      - three decades (1e2, 1e3, 1e4) so an effect that is not monotone in the magnitude still meets one rung;
        one negative rung for min-type reductions and sign handling;
      - upper bound: 1e4 x 1e3 = 1e7, so squares (1e14) and K-long dot products of two outlier tensors
        (1e14 x 4096 < 1e18) stay far below float32's 3.4e38: a correct float32 kernel never overflows.
    (One consequence, not the design target: a softmax whose max reduction misses the outlier sees exp of
    >= 95 and overflows float32.)
  * where the principle came from: looking at mutation-benchmark misses (notes/build_log.md 8e: max reductions
    that skip half a warp; tirx loops that read one element past the row). Gains on that benchmark are therefore
    tuning-set evidence only; held-out kernels / real agent revisions are the honest test.
"""
from __future__ import annotations

import torch

DISTS = ("normal", "uniform", "large", "neg_large", "offset", "tiny", "mixed", "constant", "outlier")
OUTLIER_ROW_SCALES = (1.0, 1e3)
OUTLIER_LADDER = (1e2, 1e3, -1e3, 1e4)


def parse_outlier(dist: str) -> int | None | bool:
    """'outlier' -> None (P = N), 'outlier@P' -> P, anything else -> False."""
    if dist == "outlier":
        return None
    if dist.startswith("outlier@"):
        p = int(dist.split("@", 1)[1])
        if p < 1:
            raise ValueError(f"bad period in {dist!r}: need P >= 1")
        return p
    return False


def outlier_columns(rows: int, n: int, period: int | None, gen: torch.Generator) -> torch.Tensor:
    """Column of the outlier of each row: (r mod P) + P*k, k uniform over the columns that exist; any column if
    N < P. Draws one random integer per row from gen (so it is deterministic)."""
    p = n if period is None else period
    r = torch.arange(rows)
    draw = torch.randint(0, 1 << 30, (rows,), generator=gen)
    if n >= p:
        res = r % p
        nk = (n - 1 - res) // p + 1                     # how many columns res + P*k exist in the row
        return res + p * (draw % nk)
    return draw % n


def _outlier(shape: tuple[int, ...], period: int | None, gen: torch.Generator) -> torch.Tensor:
    n = shape[-1] if shape else 1
    rows = 1
    for d in shape[:-1]:
        rows *= d
    t = torch.randn((rows, n), generator=gen, dtype=torch.float64)
    if rows == 0 or n == 0:
        return t.reshape(shape)
    r = torch.arange(rows)
    scale = torch.tensor(OUTLIER_ROW_SCALES, dtype=torch.float64)[r % len(OUTLIER_ROW_SCALES)]
    t *= scale[:, None]
    cols = outlier_columns(rows, n, period, gen)
    ladder = torch.tensor(OUTLIER_LADDER, dtype=torch.float64)[(r // 2) % len(OUTLIER_LADDER)]
    t[r, cols] = ladder * scale
    return t.reshape(shape)


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
    elif parse_outlier(dist) is not False:
        t = _outlier(tuple(shape), parse_outlier(dist), gen)
    else:
        raise ValueError(f"unknown dist {dist!r}; choose from {DISTS} (outlier also as outlier@P)")
    return t.to(torch.float32).contiguous()
