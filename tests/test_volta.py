"""Unit + integration tests for verify/ (stdlib unittest, no pytest needed).

Run:  source env.sh && envs/tirx/bin/python -m unittest discover -s tests -v
Integration tests need nvcc and the built VOLTA binary; they are skipped otherwise.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify import WORKSPACE  # noqa: E402
from verify.cases import SOFTMAX_DIR, load_manifest, softmax_case  # noqa: E402
from verify.ptx import CompileError, compile_to_ptx  # noqa: E402
from verify.volta import VOLTA_BIN, ArraySpec, VoltaCase, build_command, parse_output, run_compare  # noqa: E402

# Real VOLTA outputs captured on 2026-11-15 (volta c091559).
EQUIV_OUT = ("Exec: 0.008s  instructions: 26291  block syncs: 2176  warp syncs: 0\n"
             "VC check: 0.036s (decision procedure 0.035s)  elements: 256\nEQUIVALENT\n")
NOT_EQUIV_OUT = ("Exec: 0.007s  instructions: 23427  block syncs: 2176  warp syncs: 0\n"
                 "VC check: 0.085s (decision procedure 0.082s)  elements: 200\n"
                 "NOT EQUIVALENT: 200 mismatched element(s)\n  y[0]\n  y[1]\n  y[2]\n  ... (197 more)\n")
RACE_ERR = ("error: optimized kernel: data race on Shared[0x4]: write by t1 at pc:64 conflicts with read "
            "by t0 at pc:98\n")
LOWERING_ERR = "Error: reference kernel: lowering failed: Undefined label $L__BB0_21\n"


class ParseTests(unittest.TestCase):
    def test_equiv(self):
        r = parse_output("t", 0, EQUIV_OUT, "")
        self.assertEqual(r.status, "EQUIV")
        self.assertEqual(r.elements_checked, 256)
        self.assertEqual(r.instructions, 26291)
        self.assertAlmostEqual(r.vc_s, 0.036)

    def test_not_equiv(self):
        r = parse_output("t", 1, NOT_EQUIV_OUT, "")
        self.assertEqual(r.status, "NOT_EQUIV")
        self.assertEqual(r.mismatches, 200)
        self.assertEqual(r.mismatch_examples, ["y[0]", "y[1]", "y[2]"])

    def test_race_attributed_to_candidate(self):
        r = parse_output("t", 1, "", RACE_ERR)
        self.assertEqual(r.status, "RACE")
        self.assertEqual(r.which, "candidate")
        self.assertIn("Shared[0x4]", r.detail)

    def test_lowering_failure_is_unsupported_in_reference(self):
        r = parse_output("t", 1, "", LOWERING_ERR)
        self.assertEqual(r.status, "UNSUPPORTED")
        self.assertEqual(r.which, "reference")

    def test_other_error_kinds(self):
        cases = {
            "error: optimized kernel: deadlock: 64 thread(s) blocked": "DEADLOCK",
            "error: optimized kernel: t3: out-of-bounds Global access at 0x100400 (width 4) at pc:12": "OUT_OF_BOUNDS",
            "error: array 'y': written footprints differ (element 5 vs 6)": "FOOTPRINT",
            "error: reference kernel: t0: branch condition is symbolic at pc:9; the kernel is not a "
            "structured-CTA under this configuration": "UNSUPPORTED",
            "error: optimized kernel: unsupported at pc:3: cp.async": "UNSUPPORTED",
            "error: optimized kernel: no kernel named 'foo' in module": "CONFIG_ERROR",
            "error: output element y[3] is undefined (uninitialized read)": "UNINIT",
        }
        for err, want in cases.items():
            with self.subTest(err=err):
                self.assertEqual(parse_output("t", 1, "", err + "\n").status, want)

    def test_unrecognized_is_error(self):
        self.assertEqual(parse_output("t", 1, "", "something new\n").status, "ERROR")


class CaseTests(unittest.TestCase):
    def _case(self, **kw):
        base = dict(label="t", ref_ptx=Path("a.ptx"), ref_kernel="a", cand_ptx=Path("b.ptx"), cand_kernel="b",
                    ref_block=(1,), cand_block=(128,),
                    arrays=(ArraySpec("x", 0x1000, 4, 16, "in"), ArraySpec("y", 0x2000, 4, 16, "out")),
                    params=("ptr:x", "ptr:y", "int:16"), check_arrays=("y",))
        base.update(kw)
        return VoltaCase(**base)

    def test_command_has_block2_only_when_different(self):
        cmd = build_command(self._case())
        self.assertIn("--block2", cmd)
        cmd = build_command(self._case(cand_block=(1,)))
        self.assertNotIn("--block2", cmd)

    def test_rejects_overlapping_arrays(self):
        with self.assertRaises(ValueError):
            build_command(self._case(arrays=(ArraySpec("x", 0x1000, 4, 1024, "in"),
                                             ArraySpec("y", 0x1004, 4, 16, "out"))))

    def test_rejects_checking_an_input_array(self):
        with self.assertRaises(ValueError):
            build_command(self._case(check_arrays=("x",)))

    def test_rejects_ptr_to_unknown_array(self):
        with self.assertRaises(ValueError):
            build_command(self._case(params=("ptr:x", "ptr:z", "int:16")))


HAVE_TOOLS = shutil.which("nvcc") is not None and VOLTA_BIN.exists()


@unittest.skipUnless(HAVE_TOOLS, "needs nvcc and built VOLTA")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (WORKSPACE / "build").mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=WORKSPACE / "build")
        cls.out = Path(cls.tmp.name)
        cls.meta, cls.variants = load_manifest(SOFTMAX_DIR)
        cls.v = {v.name: v for v in cls.variants}
        cls.ptx = {n: compile_to_ptx(v.file, cls.out, mode="volta").ptx for n, v in cls.v.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, cand: str, n: int):
        ref = self.v["softmax_ref"]
        return run_compare(softmax_case(self.ptx[ref.name], ref, self.ptx[cand], self.v[cand], n))

    def test_volta_ptx_has_symexp_and_no_loc(self):
        text = self.ptx["softmax_ref"].read_text()
        self.assertIn("__symexpf", text)
        self.assertNotIn(".loc", text)

    def test_opt_equiv(self):
        self.assertEqual(self._run("softmax_opt", 200).status, "EQUIV")

    def test_nosync_race(self):
        r = self._run("softmax_bug_nosync", 200)
        self.assertEqual((r.status, r.which), ("RACE", "candidate"))

    def test_tail_bug_depends_on_shape(self):
        self.assertEqual(self._run("softmax_bug_tail", 256).status, "EQUIV")      # hidden
        self.assertEqual(self._run("softmax_bug_tail", 200).status, "NOT_EQUIV")  # exposed

    def test_compile_error_is_reported(self):
        bad = self.out / "bad.cu"
        bad.write_text('extern "C" __global__ void k(float* y) { y[0] = undefined_symbol; }\n')
        with self.assertRaises(CompileError) as ctx:
            compile_to_ptx(bad, self.out)
        self.assertIn("undefined_symbol", ctx.exception.stderr)

    def test_data_dependent_branch_is_unsupported(self):
        src = self.out / "datadep.cu"
        src.write_text('#include "common.cuh"\n'
                       'extern "C" __global__ void dd(const float* x, float* y, int N) {\n'
                       '  if (x[0] > 0.0f) y[threadIdx.x] = x[threadIdx.x]; else y[threadIdx.x] = 0.0f; }\n')
        p = compile_to_ptx(src, self.out).ptx
        case = VoltaCase(label="dd", ref_ptx=p, ref_kernel="dd", cand_ptx=p, cand_kernel="dd",
                         ref_block=(4,), cand_block=(4,),
                         arrays=(ArraySpec("x", 0x1000, 4, 4, "in"), ArraySpec("y", 0x2000, 4, 4, "out")),
                         params=("ptr:x", "ptr:y", "int:4"), check_arrays=("y",))
        r = run_compare(case)
        self.assertEqual(r.status, "UNSUPPORTED", msg=r.stderr + r.stdout)

    def test_timeout(self):
        ref = self.v["softmax_ref"]
        case = softmax_case(self.ptx[ref.name], ref, self.ptx["softmax_bug_nomax"], self.v["softmax_bug_nomax"],
                            256, timeout_s=0.01)
        self.assertEqual(run_compare(case).status, "TIMEOUT")


if __name__ == "__main__":
    unittest.main()
