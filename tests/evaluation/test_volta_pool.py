"""Parallel VOLTA for the mutant evaluation (akc/evaluation/volta_pool.py) and VOLTA shapes for tirx-lite in C.

The merge and cache tests use a fake pool (no VOLTA run); the last test runs one real VOLTA shape in one worker.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.volta_pool import PoolCache, VoltaPool, merge, superset_shapes  # noqa: E402
from akc.paths import ROOT  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation.changeaware.generate import volta_shapes  # noqa: E402
from akc.validation.interface import Candidate, Finding, ValidationContext, ValidationReport  # noqa: E402
from akc.validation.pipeline import _cache_key  # noqa: E402
from akc.validation.volta.runner import VOLTA_BIN  # noqa: E402

TXL = ROOT / "workloads" / "softmax" / "tirx"
REF = (TXL / "softmax_txl_ref.py").read_text()
BUG = (TXL / "bugs" / "softmax_txl_bug_nosync.py").read_text()


def _rep(shape, status=None, decided=True, seconds=1.0, shapeless=None):
    fs = []
    if status:
        fs.append(Finding("race", "error", "volta", f"[{shape}] VOLTA {status}", {"shape": shape, "status": status}))
    if shapeless:
        fs.append(Finding("unsupported", "info", "volta", shapeless))
    return ValidationReport("volta", not any(f.severity == "error" for f in fs), decided, fs, seconds,
                            {"runs": [{"shape": shape, "status": status or "EQUIV"}], "shapes": [shape]})


class MergeTests(unittest.TestCase):
    def test_merge_matches_the_serial_report(self):
        a, b = {"N": 256}, {"N": 584}
        m = merge([_rep(a), _rep(b, "RACE")], [a, b])
        self.assertFalse(m.passed)
        self.assertTrue(m.decided)
        self.assertEqual(m.seconds, 2.0)
        self.assertEqual([r["shape"] for r in m.artifacts["runs"]], [a, b])
        self.assertEqual(m.artifacts["shapes"], [a, b])
        self.assertEqual(len(m.errors), 1)

    def test_undecided_shape_makes_the_merge_undecided_and_shapeless_findings_appear_once(self):
        a, b = {"N": 256}, {"N": 100}
        m = merge([_rep(a, shapeless="not set up"), _rep(b, decided=False, shapeless="not set up")], [a, b])
        self.assertTrue(m.passed)
        self.assertFalse(m.decided)
        self.assertEqual(sum(f.message == "not set up" for f in m.findings), 1)


class _FakePool:
    def __init__(self):
        self.asked = []

    def report(self, cand, shapes):
        self.asked.append(shapes)
        return _rep({"N": 256})


class PoolCacheTests(unittest.TestCase):
    def test_only_volta_lookups_go_to_the_pool(self):
        parent = Candidate("softmax", "tirx", REF, label="ref")
        cand = Candidate("softmax", "tirx", BUG, parent=parent, label="bug")
        pool = _FakePool()
        cache = PoolCache(pool, cand)
        ctx = ValidationContext(device="cpu")
        self.assertIsNone(cache.get(_cache_key(cand, "fixed_tests", ctx)))
        rep = cache.get(_cache_key(cand, "volta", ctx.with_(volta_shapes=[{"N": 256}, {"N": 584}])))
        self.assertEqual(rep.validator, "volta")
        self.assertIsNotNone(cache.get(_cache_key(cand, "volta", ctx)))                  # B: default shape (None)
        self.assertEqual(pool.asked, [[{"N": 256}, {"N": 584}], None])
        cache.get(_cache_key(cand, "volta", ctx))                                       # second lookup: stored
        self.assertEqual(len(pool.asked), 2)


class ShapeTests(unittest.TestCase):
    def test_superset_contains_every_shape_C_can_ask_for(self):
        parent = Candidate("softmax", "tirx", REF, label="ref")
        cand = Candidate("softmax", "tirx", BUG, parent=parent, label="bug")
        sup = [json.dumps(s, sort_keys=True) for s in superset_shapes(cand)]
        wl = load_workload("softmax")
        self.assertEqual(json.loads(sup[0]), wl.volta["default"])
        for cats in (["sync"], ["tiling"], ["numerical"], ["indexing", "sync", "memory"]):
            from akc.validation.changeaware.generate import extract_tiles
            tiles = sorted(set(extract_tiles(BUG, "tirx")) | set(extract_tiles(REF, "tirx")))
            for s in volta_shapes("softmax", cats, tiles):
                self.assertIn(json.dumps(s, sort_keys=True), sup)


@unittest.skipUnless(Path(VOLTA_BIN).exists(), "needs the VOLTA binary")
class RealPoolTests(unittest.TestCase):
    def test_one_worker_one_shape_finds_the_missing_barrier(self):
        try:
            import tvm_tirx  # noqa: F401
        except ImportError:
            self.skipTest("needs tvm_tirx")
        parent = Candidate("softmax", "tirx", REF, label="ref")
        cand = Candidate("softmax", "tirx", BUG, parent=parent, label="pooltest_nosync")
        pool = VoltaPool(1)
        try:
            rep = pool.report(cand, None)
        finally:
            pool.shutdown()
        self.assertEqual(rep.artifacts["shapes"], [load_workload("softmax").volta["default"]])
        self.assertFalse(rep.passed, [f.message for f in rep.findings])


if __name__ == "__main__":
    unittest.main()
