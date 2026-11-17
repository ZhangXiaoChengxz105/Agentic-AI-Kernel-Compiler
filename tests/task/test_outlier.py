"""The generic outlier input family (akc/task/inputs.py) and how the change-aware generator uses it."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.task.inputs import DISTS, OUTLIER_LADDER, OUTLIER_ROW_SCALES, make_tensor  # noqa: E402
from akc.task.workload import DEFAULT_DISTS, list_workloads, load_workload  # noqa: E402
from akc.validation.changeaware.generate import floor_tests, generate_tests  # noqa: E402


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _outliers(x: torch.Tensor) -> list[tuple[int, int, float]]:
    """(row, column, value / row scale) of every element >= 50x its row scale."""
    out = []
    for r in range(x.shape[0]):
        s = OUTLIER_ROW_SCALES[r % 2]
        for c in torch.nonzero(x[r].abs() >= 50 * s).flatten().tolist():
            out.append((r, c, x[r, c].item() / s))
    return out


class OutlierTests(unittest.TestCase):
    def test_every_residue_mod_p_gets_exactly_one_outlier_per_row(self):
        for p, n in ((128, 1000), (32, 1000), (16, 16), (128, 129)):
            with self.subTest(p=p, n=n):
                x = make_tensor((p, n), f"outlier@{p}", _gen(3))
                found = _outliers(x)
                self.assertEqual([r for r, _, _ in found], list(range(p)))       # one per row
                self.assertEqual(sorted(c % p for _, c, _ in found), list(range(p)))
                self.assertTrue(all(c % p == r % p for r, c, _ in found))

    def test_row_shorter_than_the_period_still_gets_one_outlier(self):
        found = _outliers(make_tensor((20, 7), "outlier@128", _gen(1)))
        self.assertEqual([r for r, _, _ in found], list(range(20)))
        self.assertTrue(all(0 <= c < 7 for _, c, _ in found))

    def test_row_scales_alternate_and_magnitudes_follow_the_ladder(self):
        x = make_tensor((16, 2000), "outlier@128", _gen(0)).double()
        found = {r: (c, v) for r, c, v in _outliers(x)}
        for r in range(16):
            noise = torch.cat([x[r, :found[r][0]], x[r, found[r][0] + 1:]])
            self.assertAlmostEqual(noise.std().item() / OUTLIER_ROW_SCALES[r % 2], 1.0, delta=0.1)
            self.assertAlmostEqual(found[r][1], OUTLIER_LADDER[(r // 2) % 4], delta=1e-3 * abs(found[r][1]))
        ratio = x[1].abs().median() / x[0].abs().median()
        self.assertGreater(ratio.item(), 300)                                    # neighbours differ by ~1e3

    def test_deterministic_and_seed_dependent(self):
        a = make_tensor((64, 1000), "outlier@32", _gen(7))
        self.assertTrue(torch.equal(a, make_tensor((64, 1000), "outlier@32", _gen(7))))
        b = make_tensor((64, 1000), "outlier@32", _gen(8))
        self.assertNotEqual([c for _, c, _ in _outliers(a)], [c for _, c, _ in _outliers(b)])
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(torch.isfinite(a).all())

    def test_plain_outlier_and_bad_period(self):
        self.assertIn("outlier", DISTS)
        found = _outliers(make_tensor((5, 9), "outlier", _gen(0)))
        self.assertEqual([c for _, c, _ in found], [0, 1, 2, 3, 4])               # P = N: column r mod N
        with self.assertRaises(ValueError):
            make_tensor((2, 4), "outlier@0", _gen(0))

    def test_never_the_floor_stress_distribution(self):
        self.assertIn("outlier", DEFAULT_DISTS)
        self.assertNotEqual(DEFAULT_DISTS[0], "outlier")
        for op in list_workloads():
            wl = load_workload(op)
            with self.subTest(op=op):
                self.assertIn("outlier", wl.numerical_dists)
                self.assertNotEqual(wl.numerical_dists[0], "outlier")
                self.assertFalse(any(t.dist.startswith("outlier") for t in floor_tests(wl, wl.tiles)))

    def test_generator_uses_the_kernel_tile_as_period_with_enough_rows(self):
        for op, tiles in (("softmax", [128]), ("softmax", [64, 128]), ("layernorm", [128]), ("gemm", [16])):
            wl = load_workload(op)
            with self.subTest(op=op, tiles=tiles):
                outl = [t for t in generate_tests(op, ["numerical"], tiles) if t.dist.startswith("outlier")]
                self.assertEqual(len(outl), 1)
                p = min(set(tiles) | set(wl.tiles))
                self.assertEqual(outl[0].dist, f"outlier@{p}")
                rows = outl[0].shape["rows"] if "rows" in outl[0].shape else min(outl[0].shape["M"], outl[0].shape["K"])
                self.assertGreaterEqual(rows, p)
                self.assertTrue(wl.in_domain(outl[0].shape))


if __name__ == "__main__":
    unittest.main()
