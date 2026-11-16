"""Tests for ref/ (Task 1.4) and harness/ (Task 1.5/1.6).

CPU tests run anywhere (login node). GPU tests run only where torch.cuda.is_available().
Run:  source env.sh && python -m unittest tests.validation.test_runners -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.validation.runners.check import CANARY, SENTINEL_BITS, alloc_output, compare  # noqa: E402
from akc.validation.fixed_tests.suite import fixed_suite  # noqa: E402
FIXED = {op: fixed_suite(op) for op in ('softmax', 'gemm', 'layernorm')}
from akc.task.testcase import TestCase  # noqa: E402
from akc.task.inputs import DISTS, make_tensor  # noqa: E402
from akc.task.reference import LN_EPS, OPS  # noqa: E402


class InputTests(unittest.TestCase):
    def test_deterministic(self):
        for d in DISTS:
            a = make_tensor((4, 33), d, torch.Generator().manual_seed(7))
            b = make_tensor((4, 33), d, torch.Generator().manual_seed(7))
            self.assertTrue(torch.equal(a, b), d)
            self.assertEqual(a.dtype, torch.float32)

    def test_distributions_hit_their_targets(self):
        g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
        self.assertGreater(make_tensor((1, 4096), "large", g()).max().item(), 88.8)        # exp overflows in fp32
        self.assertAlmostEqual(make_tensor((1, 4096), "offset", g()).mean().item(), 1e4, delta=1)
        tiny = make_tensor((1, 4096), "tiny", g()).abs()
        self.assertTrue(((tiny > 0) & (tiny < torch.finfo(torch.float32).tiny)).any())     # subnormals
        self.assertEqual(make_tensor((2, 5), "constant", g()).unique().numel(), 1)


class GoldenTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        op = OPS["softmax"]
        y = op.golden(op.make_inputs({"rows": 3, "N": 1000}, "large", 1))
        self.assertTrue(torch.allclose(y.sum(-1), torch.ones(3, dtype=torch.float64)))
        self.assertTrue(torch.isfinite(y).all())

    def test_layernorm_matches_formula(self):
        op = OPS["layernorm"]
        i = op.make_inputs({"rows": 2, "N": 100}, "normal", 3)
        x = i["x"].double()
        want = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + LN_EPS)
        want = want * i["w"].double() + i["b"].double()
        self.assertTrue(torch.allclose(op.golden(i), want))

    def test_gemm_shapes(self):
        op = OPS["gemm"]
        s = {"M": 12, "N": 7, "K": 40}
        self.assertEqual(tuple(op.golden(op.make_inputs(s, "normal", 0)).shape), op.out_shape(s))

    def test_launch_configs(self):
        self.assertEqual(OPS["gemm"].launch({"M": 12, "N": 33, "K": 5}, (16, 16)), ((3, 1, 1), (16, 16, 1)))
        self.assertEqual(OPS["softmax"].launch({"rows": 64, "N": 1000}, (128,)), ((64, 1, 1), (128, 1, 1)))


class CheckTests(unittest.TestCase):
    def _out_from(self, values: torch.Tensor):
        buf, out = alloc_output(values.numel(), device="cpu")
        out.copy_(values.flatten().float())
        return buf, out

    def test_exact_passes(self):
        want = torch.randn(4, 8, dtype=torch.float64)
        r = compare(*self._out_from(want), want, atol=1e-6, rtol=1e-5)
        self.assertTrue(r.passed, r.message)

    def test_mismatch_reports_index(self):
        want = torch.randn(4, 8, dtype=torch.float64)
        got = want.clone()
        got[2, 5] += 1.0
        r = compare(*self._out_from(got), want, atol=1e-6, rtol=1e-5)
        self.assertEqual((r.passed, r.reason, r.n_bad), (False, "mismatch", 1))
        self.assertEqual(r.examples[0][0], [2, 5])

    def test_unwritten_detected(self):
        want = torch.zeros(10, dtype=torch.float64)
        buf, out = alloc_output(10, device="cpu")
        out[:7] = 0.0                         # last 3 never written; a zero-output check would pass
        r = compare(buf, out, want, atol=1e-6, rtol=1e-5)
        self.assertEqual((r.reason, r.n_unwritten), ("unwritten", 3))

    def test_canary_detects_out_of_bounds_store(self):
        want = torch.zeros(10, dtype=torch.float64)
        buf, out = self._out_from(want)
        buf[CANARY + 10] = 0                  # one element past the end
        self.assertEqual(compare(buf, out, want, atol=1e-6, rtol=1e-5).reason, "canary")

    def test_nan_is_nonfinite(self):
        want = torch.ones(5, dtype=torch.float64)
        got = want.clone()
        got[1] = float("nan")
        self.assertEqual(compare(*self._out_from(got), want, atol=1e-6, rtol=1e-5).reason, "nonfinite")

    def test_sentinel_is_a_nan_payload(self):
        bits = torch.tensor([SENTINEL_BITS], dtype=torch.int32)
        self.assertTrue(torch.isnan(bits.view(torch.float32)).all())

    def test_gemm_tolerance_scales_with_magnitude(self):
        op = OPS["gemm"]
        i = op.make_inputs({"M": 64, "N": 64, "K": 4096}, "normal", 0)
        got = (i["A"] @ i["B"]).double()            # float32 accumulation, like a GPU kernel
        r = compare(*self._out_from(got), op.golden(i), atol=op.atol, rtol=op.rtol, scale=op.magnitude(i))
        self.assertTrue(r.passed, r.message)


class NumericalBlindSpotEmulation(unittest.TestCase):
    """Emulate (in float32 on CPU) the two bugs VOLTA proves 'equivalent', and check the harness catches them."""

    def test_softmax_without_max_overflows(self):
        op = OPS["softmax"]
        i = op.make_inputs({"rows": 4, "N": 1000}, "large", 0)
        e = torch.exp(i["x"])                                   # bug_nomax: no max subtraction
        got = e * (1.0 / e.sum(-1, keepdim=True))
        r = compare(*self._out(got), op.golden(i), atol=op.atol, rtol=op.rtol)
        self.assertEqual(r.reason, "nonfinite")
        ok = op.make_inputs({"rows": 4, "N": 1000}, "normal", 0)  # ...but the fixed suite's data is fine
        e = torch.exp(ok["x"])
        self.assertTrue(compare(*self._out(e / e.sum(-1, keepdim=True)), op.golden(ok), atol=op.atol, rtol=op.rtol).passed)

    def test_onepass_variance_cancels(self):
        op = OPS["layernorm"]
        i = op.make_inputs({"rows": 4, "N": 1000}, "offset", 0)
        x = i["x"]
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean       # bug_onepass, float32
        got = (x - mean) * torch.rsqrt(var + LN_EPS) * i["w"] + i["b"]
        r = compare(*self._out(got), op.golden(i), atol=op.atol, rtol=op.rtol)
        self.assertFalse(r.passed)
        self.assertIn(r.reason, ("mismatch", "nonfinite"))

    def _out(self, t):
        buf, out = alloc_output(t.numel(), device="cpu")
        out.copy_(t.flatten().float())
        return buf, out


class FixedSuiteTests(unittest.TestCase):
    def test_fixed_suite_is_power_of_two_normal(self):  # shapes come from the manifests
        for op, tests in FIXED.items():
            for t in tests:
                self.assertEqual(t.dist, "normal")
                for v in t.shape.values():
                    self.assertEqual(v & (v - 1), 0, f"{op} {t.shape}")

    def test_testcase_roundtrip(self):
        t = TestCase("gemm", {"M": 12, "N": 12, "K": 40}, "large", 3, 4, "tiling")
        self.assertEqual(TestCase.from_dict(t.to_dict()), t)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU (run on a compute node)")
class GpuTests(unittest.TestCase):
    """End-to-end on a real GPU: compile (mode=gpu), launch via the driver API, check."""

    @classmethod
    def setUpClass(cls):
        from akc.backend.ptx import compile_to_ptx
        from akc.task.workload import load_workload
        cls.v, cls.ptx = {}, {}
        out = Path(__file__).resolve().parents[2] / "build" / "ptx_gpu_test"
        for op in ("softmax", "gemm", "layernorm"):
            for v in [x for x in load_workload(op).variants if x.lang == "cuda"]:
                cls.v[v.name] = (op, v)
                cls.ptx[v.name] = compile_to_ptx(v.file, out / op, mode="gpu").ptx

    def _run(self, name, tests):
        from akc.validation.runners.gpu import run_tests_isolated
        op, v = self.v[name]
        return run_tests_isolated(op, "cuda", self.ptx[name], name, v.block, tests)

    def test_reference_and_opt_pass_fixed_suite(self):
        for op in ("softmax", "gemm", "layernorm"):
            for name in (f"{op}_ref", f"{op}_opt"):
                for r in self._run(name, FIXED[op]):
                    self.assertTrue(r["passed"], f"{name}: {r['message']}")

    def test_blind_spots_caught_on_gpu(self):
        r = self._run("softmax_bug_nomax", [TestCase("softmax", {"rows": 8, "N": 1000}, "large")])[0]
        self.assertEqual(r["reason"], "nonfinite", r["message"])
        r = self._run("layernorm_bug_onepass", [TestCase("layernorm", {"rows": 8, "N": 1000}, "offset")])[0]
        self.assertFalse(r["passed"], r["message"])

    def test_shape_bugs_caught_on_gpu(self):
        r = self._run("softmax_bug_tail", [TestCase("softmax", {"rows": 8, "N": 1000})])[0]
        self.assertEqual(r["reason"], "mismatch", r["message"])
        r = self._run("gemm_bug_guard", [TestCase("gemm", {"M": 100, "N": 100, "K": 40})])[0]
        self.assertFalse(r["passed"], r["message"])


if __name__ == "__main__":
    unittest.main()
