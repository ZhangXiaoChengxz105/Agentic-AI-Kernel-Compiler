"""Validator interface, pipeline (configurations A/B/C), feedback rendering and the hottest CLI.

CPU-only by default (dynamic CUDA tests are reported "skipped" without a GPU; VOLTA runs on the CPU).
Set AKC_SLOW=1 to also run the NumSim (tirx on CPU) test (~1-2 min: builds a native simulator artifact).
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.feedback.render import render  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation import Candidate, Finding, ValidationContext, ValidationReport, run_pipeline  # noqa: E402
from akc.validation.changeaware.generate import extract_tiles, generate_tests  # noqa: E402
from akc.validation.changeaware.validator import ChangeAwareValidator  # noqa: E402
from akc.validation.volta.runner import VOLTA_BIN  # noqa: E402
from akc.validation.volta.validator import VoltaValidator  # noqa: E402

HAVE_VOLTA = shutil.which("nvcc") is not None and VOLTA_BIN.exists()
CPU = ValidationContext(device="cpu")


def _cand(op: str, name: str, parent: str | None = None) -> Candidate:
    wl = load_workload(op)
    v = wl.variant(name)
    par = None
    if parent:
        p = wl.variant(parent)
        par = Candidate.from_file(op, p.file, entry=p.name, label=p.name)
    return Candidate.from_file(op, v.file, entry=v.name if v.lang == "cuda" else None, block=v.block, parent=par, label=v.name)


class InterfaceTests(unittest.TestCase):
    def test_finding_validates_kind_and_severity(self):
        with self.assertRaises(AssertionError):
            Finding("made_up", "error", "x", "m")
        with self.assertRaises(AssertionError):
            Finding("race", "fatal", "x", "m")

    def test_report_passed_follows_error_findings(self):
        import time
        ok = ValidationReport.build("v", [Finding("unsupported", "info", "v", "m")], False, time.perf_counter())
        bad = ValidationReport.build("v", [Finding("race", "error", "v", "m")], True, time.perf_counter())
        self.assertEqual((ok.passed, bad.passed), (True, False))

    def test_candidate_materialize_writes_source(self):
        c = Candidate("softmax", "cuda", "// hello\n", label="mat_test")
        p = c.materialize()
        self.assertEqual(p.read_text(), "// hello\n")

    def test_tirx_candidate_is_unsupported_for_volta(self):
        r = VoltaValidator().validate(_cand("softmax", "softmax_txl_ref"), CPU)
        self.assertEqual((r.passed, r.decided, r.findings[0].kind), (True, False, "unsupported"))


class GeneratorFeatureTests(unittest.TestCase):
    def test_extract_tiles(self):
        self.assertEqual(extract_tiles("#define TILE 32\n#define BLOCK 128\n", "cuda"), [32, 128])
        self.assertEqual(extract_tiles("BLOCK = 256\nx = 3\n", "tirx"), [256])
        self.assertEqual(extract_tiles("constexpr int BM = 64;", "cuda"), [64])

    def test_candidate_tiles_drive_boundaries(self):
        shapes = [t.shape for t in generate_tests("gemm", ["tiling"], [32])]
        self.assertTrue(any(s["K"] % 32 and s["M"] < 32 for s in shapes))      # sub-tile matrix for T=32

    def test_memory_tests_include_misaligned_pointers(self):
        self.assertTrue(any(t.misalign for t in generate_tests("softmax", ["memory"], None, budget=20)))

    def test_sync_tests_include_grid_sweep(self):
        rows = {t.shape["rows"] for t in generate_tests("softmax", ["sync"], None, budget=20)}
        self.assertTrue({1, 2048} <= rows)


class ChangeAwareValidatorTests(unittest.TestCase):
    def test_cpu_run_reports_plan_and_skips_cuda_tests(self):
        r = ChangeAwareValidator().validate(_cand("softmax", "softmax_bug_tail", "softmax_opt"), CPU)
        self.assertEqual(r.artifacts["categories"][0], "tiling")
        self.assertFalse(r.decided)                                   # nothing executed without a GPU
        self.assertTrue(all(f.kind == "skipped" for f in r.findings))
        self.assertTrue(any(s["N"] % 128 for s in r.artifacts["suggested_volta_shapes"]))

    def test_parent_defaults_to_reference(self):
        r = ChangeAwareValidator().validate(_cand("softmax", "softmax_opt"), CPU)
        self.assertEqual(r.artifacts["parent"], "reference softmax_ref")


@unittest.skipUnless(HAVE_VOLTA, "needs nvcc and VOLTA")
class PipelineTests(unittest.TestCase):
    def test_config_B_misses_and_C_catches_the_tail_bug(self):
        cand = _cand("softmax", "softmax_bug_tail", "softmax_opt")
        b = run_pipeline(cand, "B", CPU)
        c = run_pipeline(cand, "C", CPU)
        self.assertTrue(b.passed)                       # fixed tests skipped (no GPU) + VOLTA default shape misses
        self.assertFalse(c.passed)
        self.assertIn("not_equivalent", {f.kind for f in c.findings})
        self.assertEqual([r.validator for r in c.reports], ["fixed_tests", "changeaware", "volta"])

    def test_volta_race_and_hint_rendering(self):
        r = VoltaValidator().validate(_cand("softmax", "softmax_bug_nosync"), CPU)
        self.assertEqual(r.findings[0].kind, "race")
        text = render([r])
        self.assertIn("Validation FAILED", text)
        self.assertIn("hint:", text)

    def test_compile_error_stops_pipeline(self):
        cand = Candidate("softmax", "cuda", 'extern "C" __global__ void k(float* y) { y[0] = nope; }\n', label="broken")
        res = run_pipeline(cand, "C", CPU)
        self.assertFalse(res.passed)
        self.assertEqual(res.reports[-1].findings[0].kind, "compile_error")

    def test_hottest_cli(self):
        from akc.validation.changeaware.hottest import main
        wl = load_workload("gemm")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--workload", "gemm", "--new", str(wl.variant("gemm_bug_guard").file),
                         "--old", str(wl.variant("gemm_opt").file), "--device", "cpu"])
        out = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("change categories: indexing", out)
        self.assertIn("Validation FAILED", out)


@unittest.skipUnless(os.environ.get("AKC_SLOW") == "1", "set AKC_SLOW=1 (NumSim builds a native artifact, ~1-2 min)")
class NumSimTests(unittest.TestCase):
    def test_tirx_offbyone_caught_on_cpu(self):
        from akc.task.testcase import TestCase
        from akc.validation.fixed_tests.validator import FixedTestsValidator
        t = [TestCase("softmax", {"rows": 2, "N": 200}, "normal", 0, 1, "slow-test")]
        r = FixedTestsValidator().validate(_cand("softmax", "softmax_txl_bug_offbyone"), CPU.with_(test_cases=t))
        self.assertFalse(r.passed, r.findings)
        self.assertEqual(r.findings[0].kind, "out_of_bounds")


if __name__ == "__main__":
    unittest.main()
