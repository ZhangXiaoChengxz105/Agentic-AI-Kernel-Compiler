"""Tests for akc.perf (speed objective) and feedback.render_perf.

CPU tests check the roofline arithmetic against hand-computed numbers and the feedback text; GPU tests
(only where torch.cuda.is_available()) time real kernels.
Run:  source env.sh && python -m unittest tests.perf.test_perf -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.feedback.render import render_observation, render_perf  # noqa: E402
from akc.perf import PerfReport, analyze, measure, spec_for  # noqa: E402
from akc.task.workload import list_workloads, load_workload  # noqa: E402
from akc.validation.interface import Candidate, ValidationReport  # noqa: E402

A100 = spec_for("NVIDIA A100-SXM4-40GB")
SOFTMAX = {"rows": 4096, "N": 1024}
GEMM = {"M": 1024, "N": 1024, "K": 1024}
GPU = torch.cuda.is_available()


class HardwareTests(unittest.TestCase):
    def test_a100_datasheet(self):
        self.assertEqual(A100.dram_gbps, 1555.0)
        self.assertEqual(A100.fp32_tflops, 19.5)
        self.assertAlmostEqual(A100.ridge, 12.54, places=2)          # FLOP/byte
        self.assertTrue(A100.known)
        self.assertEqual(spec_for("NVIDIA A100-SXM4-80GB").dram_gbps, 2039.0)

    def test_unknown_gpu_falls_back_and_says_so(self):
        s = spec_for("Some Future GPU")
        self.assertFalse(s.known)
        self.assertEqual(s.dram_gbps, A100.dram_gbps)


class RooflineTests(unittest.TestCase):
    def test_softmax_numbers_from_the_a100_run(self):
        # notes/gpu_results.md: softmax_opt 0.0363 ms, PyTorch 0.0370 ms at rows=4096, N=1024
        r = analyze("softmax", SOFTMAX, {"median_ms": 0.0363, "iters": 50}, A100, torch_ms=0.0370, copy_ms=0.027)
        self.assertTrue(r.measured)
        self.assertEqual(r.bytes, 8 * 4096 * 1024)                    # read x once, write y once
        self.assertAlmostEqual(r.achieved_gbps, 924.4, delta=0.5)
        self.assertAlmostEqual(r.pct_peak_bw, 59.4, delta=0.1)
        self.assertEqual(r.bound, "memory")
        self.assertAlmostEqual(r.speedup_vs_torch, 0.0370 / 0.0363)
        self.assertEqual(r.floor_ms, 0.027)                           # copy time is the practical floor
        self.assertAlmostEqual(r.headroom, 0.0363 / 0.027)
        self.assertLess(r.roofline_ms, r.floor_ms)                    # datasheet bound is below the copy

    def test_gemm_is_compute_bound(self):
        r = analyze("gemm", GEMM, {"median_ms": 0.609}, A100, torch_ms=0.172, copy_ms=0.01)
        self.assertEqual(r.bound, "compute")
        self.assertAlmostEqual(r.achieved_tflops, 3.526, delta=0.01)
        self.assertAlmostEqual(r.pct_peak_flops, 18.1, delta=0.1)
        self.assertIsNone(r.copy_ms)                                  # copy floor is meaningless when compute-bound
        self.assertAlmostEqual(r.floor_ms, 2 * 1024 ** 3 / 19.5e12 * 1e3)

    def test_tiny_kernels_are_latency_bound(self):
        r = analyze("softmax", {"rows": 4, "N": 32}, {"median_ms": 0.004}, A100)
        self.assertEqual(r.bound, "latency")

    def test_report_roundtrip_and_best(self):
        a = analyze("softmax", SOFTMAX, {"median_ms": 0.04}, A100, label="r1")
        b = analyze("softmax", SOFTMAX, {"median_ms": 0.03}, A100, label="r2")
        skipped = PerfReport.not_measured("softmax", "r3", "cuda", SOFTMAX, "no GPU")
        self.assertTrue(b.faster_than(a))
        self.assertFalse(a.faster_than(b))
        self.assertTrue(a.faster_than(None))
        self.assertTrue(a.faster_than(skipped))
        self.assertFalse(skipped.faster_than(a))
        d = b.to_dict()
        self.assertIn("speedup_vs_torch", d)
        self.assertEqual(PerfReport.from_dict(d), b)


class RenderTests(unittest.TestCase):
    def test_memory_bound_text(self):
        r = analyze("softmax", SOFTMAX, {"median_ms": 0.0363, "iters": 50}, A100, torch_ms=0.0370, copy_ms=0.027)
        t = render_perf(r)
        for s in ("0.0363 ms", "1.02x PyTorch", "Memory-bound", "59% of the 1555 GB/s peak", "practical floor", "1.34x"):
            self.assertIn(s, t)

    def test_compute_bound_text(self):
        t = render_perf(analyze("gemm", GEMM, {"median_ms": 0.609}, A100, torch_ms=0.172))
        for s in ("0.28x PyTorch", "Compute-bound", "18% of the 19.5 TFLOP/s FP32 peak", "PyTorch reaches 12.49"):
            self.assertIn(s, t)

    def test_best_so_far(self):
        best = analyze("softmax", SOFTMAX, {"median_ms": 0.0343}, A100, label="r1", torch_ms=0.037)
        slower = analyze("softmax", SOFTMAX, {"median_ms": 0.0407}, A100, label="r3", torch_ms=0.037)
        self.assertIn("Best correct version so far: r1", render_perf(slower, best))
        self.assertIn("New best correct version", render_perf(best, slower))

    def test_not_measured_and_observation(self):
        r = PerfReport.not_measured("softmax", "x", "cuda", SOFTMAX, "no GPU here")
        self.assertEqual(render_perf(r), "Speed not measured (no GPU here).")
        obs = render_observation([ValidationReport("fixed_tests", True, True)], r)
        self.assertTrue(obs.startswith("All applicable checks passed"))
        self.assertIn("Speed not measured", obs)


class MeasureTests(unittest.TestCase):
    def test_every_workload_has_a_perf_shape(self):
        for w in list_workloads():
            wl = load_workload(w)
            self.assertEqual(sorted(wl.perf_shape), sorted(wl.shape_keys), w)

    @unittest.skipIf(GPU, "CPU-only behaviour")
    def test_no_gpu_is_reported_not_raised(self):
        wl = load_workload("softmax")
        r = measure(Candidate.from_file("softmax", wl.variant("softmax_opt").file, entry="softmax_opt"))
        self.assertFalse(r.measured)
        self.assertIn("no GPU", r.reason)

    def test_shape_outside_the_domain_is_rejected(self):
        from unittest import mock

        from akc.task import workload as wmod
        wl = load_workload("softmax")
        cand = Candidate.from_file("softmax", wl.variant("softmax_opt").file, entry="softmax_opt")
        for shape in ({"rows": 4096, "N": 16385}, {"rows": 4097, "N": 1024}):
            r = measure(cand, shape, device="gpu")                       # rejected before any GPU work
            self.assertFalse(r.measured)
            self.assertIn("outside the declared shape domain", r.reason)
            self.assertIn("N in [1, 16384]", r.reason)
        bare = wmod.Workload(wl.op, wl.dir, {k: v for k, v in wl.meta.items() if k != "domain"}, wl.variants)
        with mock.patch.object(wmod, "load_workload", return_value=bare):      # no domain: not checked (as before)
            r = measure(cand, {"rows": 4096, "N": 16385}, device="cpu")
        self.assertNotIn("domain", r.reason)

    def test_compile_error_is_reported(self):
        c = Candidate("softmax", "cuda", 'extern "C" __global__ void k(float* x) { this is not C }', label="broken")
        r = measure(c, device="gpu")
        self.assertFalse(r.measured)
        self.assertIn("compile error", r.reason)

    @unittest.skipUnless(GPU, "needs a GPU")
    def test_gpu_softmax_opt(self):
        wl = load_workload("softmax")
        r = measure(Candidate.from_file("softmax", wl.variant("softmax_opt").file, entry="softmax_opt"), iters=20)
        self.assertTrue(r.measured, r.reason)
        self.assertEqual(r.bound, "memory")
        self.assertGreater(r.speedup_vs_torch, 0.5)
        self.assertTrue(5 < r.pct_peak_bw < 100, r.pct_peak_bw)
        self.assertGreater(r.headroom, 0.8)
        self.assertTrue(spec_for(r.gpu).known, r.gpu)

    @unittest.skipUnless(GPU, "needs a GPU")
    def test_gpu_wrong_kernel_is_not_timed(self):
        wl = load_workload("softmax")
        bug = Candidate.from_file("softmax", wl.variant("softmax_bug_tail").file, entry="softmax_bug_tail")
        r = measure(bug, {"rows": 64, "N": 1000}, iters=5)          # N not a multiple of 128: the tail bug shows
        self.assertFalse(r.measured)
        self.assertIn("incorrect at the benchmark shape", r.reason)


if __name__ == "__main__":
    unittest.main()
