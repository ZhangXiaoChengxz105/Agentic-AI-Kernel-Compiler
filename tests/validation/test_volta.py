"""Unit + integration tests for verify/ (stdlib unittest, no pytest needed).

Run:  source env.sh && python -m unittest discover -s tests -v
Integration tests need nvcc and the built VOLTA binary; they are skipped otherwise.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.paths import ROOT as WORKSPACE  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation.volta.layouts import expected_statuses, make_case, place_arrays  # noqa: E402
from akc.backend.ptx import CompileError, compile_to_ptx  # noqa: E402
from akc.validation.volta.runner import VOLTA_BIN, ArraySpec, VoltaCase, build_command, parse_output, run_compare  # noqa: E402

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
            # real output (GLM pilot, 2026-11-15): kernel used -FLT_MAX (3.4e38 > i128 max 1.7e38)
            "error: equivalence check failed: rational coefficient overflowed i128": "UNSUPPORTED",
            # real output (softmax_bug_tail at N=100: empty reduction -> sum identically 0)
            "error: equivalence check failed: division by a symbolically zero value": "DIV_BY_ZERO",
            "error: equivalence check failed: VC too large: exceeded 4000000000 term operations": "UNSUPPORTED",
            "error: equivalence check failed: cannot check equivalence of undefined values": "UNINIT",
        }
        for err, want in cases.items():
            with self.subTest(err=err):
                self.assertEqual(parse_output("t", 1, "", err + "\n").status, want)

    def test_unrecognized_is_error(self):
        self.assertEqual(parse_output("t", 1, "", "something new\n").status, "ERROR")


class ExpectationTests(unittest.TestCase):
    def test_rule_formats(self):
        rules = [{"if": "K % 16 == 0", "then": "EQUIV"}, {"then": "NOT_EQUIV"}]
        self.assertEqual(expected_statuses(rules, {"K": 48}), ("EQUIV",))
        self.assertEqual(expected_statuses(rules, {"K": 40}), ("NOT_EQUIV",))
        self.assertEqual(expected_statuses("RACE", {}), ("RACE",))
        self.assertEqual(expected_statuses(["RACE", "OUT_OF_BOUNDS"], {}), ("RACE", "OUT_OF_BOUNDS"))
        self.assertIsNone(expected_statuses(None, {}))

    def test_arrays_never_overlap(self):
        arrs = place_arrays([("a", 4, 1000, "in"), ("b", 4, 70000, "in"), ("c", 4, 1, "out")])
        for x, y in zip(arrs, arrs[1:]):
            self.assertLess(x.end, y.base)


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
        cls.v, cls.ptx = {}, {}
        for op in ("softmax", "gemm", "layernorm"):
            for v in load_workload(op).variants:
                if v.lang != "cuda":
                    continue
                cls.v[v.name] = (op, v)
                cls.ptx[v.name] = compile_to_ptx(v.file, cls.out / op, mode="volta").ptx

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, cand: str, shape: dict, timeout_s: float = 600.0):
        op, c = self.v[cand]
        ref = self.v[f"{op}_ref"][1]
        return run_compare(make_case(op, self.ptx[ref.name], ref, self.ptx[cand], c, shape, timeout_s))

    def test_volta_ptx_has_symexp_and_no_loc(self):
        text = self.ptx["softmax_ref"].read_text()
        self.assertIn("__symexpf", text)
        self.assertNotIn(".loc", text)

    def test_opt_equiv(self):
        self.assertEqual(self._run("softmax_opt", {"N": 200}).status, "EQUIV")

    def test_nosync_race(self):
        r = self._run("softmax_bug_nosync", {"N": 200})
        self.assertEqual((r.status, r.which), ("RACE", "candidate"))

    def test_tail_bug_depends_on_shape(self):
        self.assertEqual(self._run("softmax_bug_tail", {"N": 256}).status, "EQUIV")      # hidden
        self.assertEqual(self._run("softmax_bug_tail", {"N": 200}).status, "NOT_EQUIV")  # exposed

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
        self.assertEqual(self._run("softmax_bug_nomax", {"N": 256}, timeout_s=0.01).status, "TIMEOUT")

    def test_gemm_guard_bug_needs_edge_in_cta0(self):
        self.assertEqual(self._run("gemm_bug_guard", {"M": 32, "N": 32, "K": 40}).status, "EQUIV")  # hidden
        self.assertIn(self._run("gemm_bug_guard", {"M": 12, "N": 12, "K": 40}).status,
                      ("RACE", "OUT_OF_BOUNDS", "FOOTPRINT", "NOT_EQUIV"))

    def test_gemm_ktail_and_layernorm(self):
        self.assertEqual(self._run("gemm_bug_ktail", {"M": 12, "N": 12, "K": 40}).status, "NOT_EQUIV")
        self.assertEqual(self._run("layernorm_opt", {"N": 100}).status, "EQUIV")
        self.assertEqual(self._run("layernorm_bug_onepass", {"N": 100}).status, "EQUIV")   # blind spot
        self.assertEqual(self._run("layernorm_bug_nosync", {"N": 100}).status, "RACE")


# ---- tirx-lite path (akc/validation/volta/tirx.py, layouts.make_tirx_case, VoltaValidator._validate_tirx) ----

from akc.validation.volta import tirx as vtirx  # noqa: E402
from akc.validation.volta.layouts import make_tirx_case  # noqa: E402
from akc.validation.volta.runner import build_analyze_command, parse_analyze_output  # noqa: E402

# Excerpts of TIRx's generated CUDA (tvm-tirx 0.1.0, 2026-11-17).
EX2_HELPER = ("__forceinline__ __device__ void tvm_builtin_ptx_ex2_approx_f32(float& __d, float __value) {\n"
              '  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(__d) : "f"(__value));\n}\n')
BAR_HELPER = ("__forceinline__ __device__ void tvm_builtin_ptx_bar_sync_count_sync(uint32_t __a, uint32_t __b) {\n"
              '  asm volatile("bar.sync %0, %1;" :  : "r"(__a), "r"(__b) : "memory");\n}\n')
KERNEL_HEAD = ('extern "C" __global__ void __launch_bounds__(128) softmax_txl_kernel(float* __restrict__ x_ptr, '
               'float* __restrict__ y_ptr) {\n')


class TirxRewriteTests(unittest.TestCase):
    def test_expf_becomes_symexpf(self):
        src = KERNEL_HEAD + "  a = expf((x - m)); b = __expf(y); c = __symexpf(z); d = myexpf(w);\n}\n"
        out, n = vtirx.volta_rewrite(src, ("exp",))
        self.assertEqual(n, {"exp": 2})
        self.assertIn("a = __symexpf((x - m))", out)
        self.assertIn("b = __symexpf(y)", out)
        self.assertIn("c = __symexpf(z)", out)        # untouched
        self.assertIn("d = myexpf(w)", out)          # not exp
        self.assertIn('extern "C" __device__ float __symexpf(float x);', out)   # prelude added once
        self.assertLess(out.index("__symexpf(float x)"), out.index("__global__"))

    def test_ex2_is_read_as_exp_over_log2e(self):
        src = EX2_HELPER + KERNEL_HEAD + "  t = exp2f(u);\n}\n"
        out, n = vtirx.volta_rewrite(src, ("ex2",))
        self.assertEqual(n, {"ex2": 2})
        self.assertIn("__d = __volta_ex2(__value);", out)
        self.assertIn("t = __volta_ex2(u)", out)
        self.assertNotIn("ex2.approx", out)
        self.assertIn(f"__symexpf(t / {vtirx.LOG2E_F32})", out)
        self.assertEqual(float.fromhex(vtirx.LOG2E_F32[:-1]), 1.44269502162933349609375)   # fl32(log2 e)

    def test_huge_constants_become_infinity(self):
        src = (KERNEL_HEAD + "  m = -0x1.fffffep+127f/*-3.402823e+38*/;\n  k = -0x1.ff933c78cdfadp+127f;\n"
               "  ok = 0x1.fffffep+126f;\n  d = 3.4e+38f;\n  e = 1.5e+38f;\n"
               "    uint v_ = (uint)4286578687;\n  r = (*(float *)(&(v_)));\n"      # -FLT_MAX bits, read as float
               "    uint v__1 = (uint)4286578688;\n  q = (*(float *)(&(v__1)));\n"  # already -inf: unchanged
               "    uint v__2 = (uint)4286578687;\n  w = v__2 + 1;\n}\n")             # used as an integer: unchanged
        out, n = vtirx.volta_rewrite(src, ("huge_const",))
        self.assertEqual(n, {"huge_const": 4})
        self.assertIn("m = -__VOLTA_INF;", out)
        self.assertIn("k = -__VOLTA_INF;", out)
        self.assertIn("ok = 0x1.fffffep+126f;", out)  # 2^126 * 1.99 < 2^127: VOLTA can represent it
        self.assertIn("d = __VOLTA_INF;", out)
        self.assertIn("e = 1.5e+38f;", out)
        self.assertIn("uint v_ = (uint)4286578688;", out)
        self.assertIn("uint v__1 = (uint)4286578688;", out)
        self.assertIn("uint v__2 = (uint)4286578687;", out)

    def test_full_cta_counted_barrier_becomes_bar_sync(self):
        src = BAR_HELPER + KERNEL_HEAD + ("  tvm_builtin_ptx_bar_sync_count_sync((uint)0, (uint)128);\n"
                                          "  tvm_builtin_ptx_bar_sync_count_sync((uint)1, (uint)64);\n}\n")
        out, n = vtirx.volta_rewrite(src, ("bar_count",), nthreads=128)
        self.assertEqual(n, {"bar_count": 1})
        self.assertIn('asm volatile("bar.sync 0;" ::: "memory");', out)
        self.assertIn("tvm_builtin_ptx_bar_sync_count_sync((uint)1, (uint)64);", out)   # partial CTA: kept

    def test_no_rewrite_no_prelude(self):
        src = KERNEL_HEAD + "  y = x;\n}\n"
        out, n = vtirx.volta_rewrite(src)
        self.assertEqual(out, src)
        self.assertFalse(any(n.values()))
        with self.assertRaises(ValueError):
            vtirx.volta_rewrite(src, ("nope",))

    def test_select_to_minmax_on_ptx(self):
        ptx = ("\tsetp.lt.f32 \t%p2, %f11, %f9;\n\tselp.f32 \t%f12, %f9, %f11, %p2;\n"     # f11 < f9 ? f9 : f11
               "\tsetp.gt.f32 \t%p3, %f12, %f13;\n\tselp.f32 \t%f14, %f13, %f12, %p3;\n"   # f12 > f13 ? f13 : f12
               # txl.max's NaN-propagating form: (a > b || a != a) ? a : b
               "\tsetp.gt.f32 \t%p4, %f5, %f3;\n\tsetp.nan.f32 \t%p5, %f5, %f5;\n"
               "\tor.pred \t%p6, %p4, %p5;\n\tselp.f32 \t%f4, %f5, %f3, %p6;\n"
               # predicate also used by a branch: left alone
               "\tsetp.lt.f32 \t%p7, %f1, %f2;\n\tselp.f32 \t%f6, %f2, %f1, %p7;\n@%p7 bra \t$L__BB0_2;\n"
               # operands do not match the comparison: left alone
               "\tsetp.lt.f32 \t%p8, %f1, %f2;\n\tselp.f32 \t%f7, %f2, %f9, %p8;\n")
        out, n = vtirx.ptx_select_to_minmax(ptx)
        self.assertEqual(n, 3)
        self.assertIn("max.f32 \t%f12, %f11, %f9;", out)
        self.assertIn("min.f32 \t%f14, %f12, %f13;", out)
        self.assertIn("max.f32 \t%f4, %f5, %f3;", out)
        self.assertIn("selp.f32 \t%f6, %f2, %f1, %p7;", out)
        self.assertIn("selp.f32 \t%f7, %f2, %f9, %p8;", out)

    def test_parse_entry(self):
        name, params = vtirx.parse_entry(BAR_HELPER + KERNEL_HEAD + "}\n")
        self.assertEqual((name, params), ("softmax_txl_kernel", [("ptr", "x"), ("ptr", "y")]))
        _, params = vtirx.parse_entry('extern "C" __global__ void k(float* __restrict__ a_ptr, int n) {}')
        self.assertEqual(params, [("ptr", "a"), ("scalar", "int n")])
        with self.assertRaises(vtirx.TirxBuildError):
            vtirx.parse_entry("__global__ void k() {}")


class TirxCaseTests(unittest.TestCase):
    def _b(self, label, smem=512, params=("x", "y"), grid=(2,), block=(128,)):
        return vtirx.TirxPtx(label, Path(f"{label}.ptx"), Path(f"{label}.cu"), f"{label}_kernel", grid, block,
                             smem, params)

    def test_case_from_two_builds(self):
        case = make_tirx_case("softmax", self._b("ref", 512), self._b("cand", 1040), {"N": 200})
        self.assertEqual([(a.name, a.length, a.kind) for a in case.arrays], [("x", 400, "in"), ("y", 400, "out")])
        self.assertEqual(case.params, ("ptr:x", "ptr:y"))           # static shapes: no int:N
        self.assertEqual(case.check_arrays, ("y",))
        self.assertEqual(case.dyn_shared, 1040)
        cmd = build_command(case)
        self.assertEqual(cmd[cmd.index("--dyn-shared") + 1], "1040")
        self.assertEqual(cmd[cmd.index("-g") + 1], "2")

    def test_rejects_different_parameter_order(self):
        with self.assertRaises(ValueError):
            make_tirx_case("softmax", self._b("ref"), self._b("cand", params=("y", "x")), {"N": 200})

    def test_analyze_command_uses_the_candidates_own_smem(self):
        case = make_tirx_case("softmax", self._b("ref", 512), self._b("cand", 16), {"N": 256})
        cmd = build_analyze_command(case, 16)
        self.assertEqual(cmd[2:5], ["analyze", "cand.ptx", "-k"])
        self.assertEqual(cmd[cmd.index("--dyn-shared") + 1], "16")
        self.assertNotIn("--check-array", cmd)

    def test_parse_analyze_output(self):
        ok = parse_analyze_output("t", 0, "Analysis complete: no data races or deadlocks detected.\n", "")
        self.assertEqual((ok.status, ok.which), ("CLEAN", "candidate"))
        bad = parse_analyze_output("t", 1, "", "analysis failed: t64: out-of-bounds Shared access at 0x100 "
                                               "(width 4) at pc:27\n")
        self.assertEqual((bad.status, bad.which), ("OUT_OF_BOUNDS", "candidate"))
        self.assertTrue(bad.detail.startswith("t64:"))
        # real output (GLM kernel C before the select_max rewrite)
        self.assertEqual(parse_output("t", 1, "", "error: equivalence check failed: non-finite constant -inf in "
                                                  "expression\n").status, "UNSUPPORTED")

    def test_specific_hints_win_over_lowering_failed(self):
        from akc.validation.volta.hints import hint_for
        # real outputs (ablation job 24060724: a rewrite switched off)
        bar = parse_output("t", 1, "", "Error: optimized kernel: lowering failed: Unsupported instruction: bar.sync\n"
                                       "Reason: register barrier id (Ident(AsciiString { data: ['%', 'r'] }))\n")
        self.assertEqual(bar.status, "UNSUPPORTED")
        self.assertIn("literal constant", hint_for(bar, "tirx"))
        ex2 = parse_output("t", 1, "", "Error: optimized kernel: lowering failed: Unsupported instruction: "
                                       "Ex2(Float32 { ftz: false })\n")
        self.assertIn("txl.exp", hint_for(ex2, "tirx"))
        self.assertIn("EXP(x)", hint_for(ex2, "cuda"))
        other = parse_output("t", 1, "", "Error: optimized kernel: lowering failed: Unsupported instruction: cp.async\n")
        self.assertIn("cp.async", hint_for(other, "tirx"))
        self.assertIn("Avoid inline PTX", hint_for(other, "cuda"))

    def test_workload_without_tirx_reference_is_unsupported(self):
        # since Task 1 (2026-11-17) every workload has a tirx reference, so the manifest's tirx parts are stripped
        from unittest import mock

        from akc.task.workload import Workload
        from akc.validation.interface import Candidate, ValidationContext
        from akc.validation.volta import validator as vmod
        wl = load_workload("gemm")
        meta = {k: v for k, v in wl.meta.items() if k != "tirx"}
        meta["references"] = {k: v for k, v in meta["references"].items() if k != "tirx"}
        with mock.patch.object(vmod, "load_workload", return_value=Workload(wl.op, wl.dir, meta, wl.variants)):
            rep = vmod.VoltaValidator().validate(Candidate("gemm", "tirx", "x = 1\n", label="t"), ValidationContext())
        self.assertFalse(rep.decided)
        self.assertEqual([f.kind for f in rep.findings], ["unsupported"])


def _have_tirx() -> bool:
    try:
        import tirx_kernels.tirx_lite  # noqa: F401
        import tvm_tirx  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(HAVE_TOOLS and _have_tirx(), "needs nvcc, built VOLTA and tvm_tirx")
class TirxIntegrationTests(unittest.TestCase):
    """End to end through VoltaValidator (trace -> generated CUDA -> rewrites -> PTX -> volta compare), N <= 256."""

    @classmethod
    def setUpClass(cls):
        from akc.validation.volta.validator import VoltaValidator
        cls.validator = VoltaValidator(timeout_s=120)
        cls.tirx = WORKSPACE / "workloads" / "softmax" / "tirx"

    def _run(self, src: str, label: str, n: int):
        from akc.validation.interface import Candidate, ValidationContext
        rep = self.validator.validate(Candidate("softmax", "tirx", src, label=label),
                                      ValidationContext(volta_shapes=[{"N": n}]))
        return rep, [r["status"] for r in rep.artifacts["runs"]]

    def test_reference_equiv_and_all_rewrites_used(self):
        rep, st = self._run((self.tirx / "softmax_txl_ref.py").read_text(), "ref", 200)
        self.assertEqual(st, ["EQUIV"])
        self.assertTrue(rep.passed and rep.decided)
        rw = rep.artifacts["runs"][0]["rewrites"]["candidate"]
        self.assertEqual(rw["exp"], 2)
        self.assertEqual(rw["huge_const"], 1)         # NEG_BIG = -3.4e38
        self.assertGreater(rw["select_max"], 0)       # txl.max
        ptx = Path(rep.artifacts["runs"][0]["ptx"]["candidate"]).read_text()
        self.assertIn("call.uni", ptx)
        self.assertNotIn(".loc", ptx)
        self.assertNotIn("ex2.approx", ptx)

    def test_injected_bugs(self):
        bugs = self.tirx / "bugs"
        rep, st = self._run((bugs / "softmax_txl_bug_nosync.py").read_text(), "nosync", 128)
        self.assertEqual(st, ["RACE"])
        self.assertEqual(rep.findings[0].kind, "race")
        self.assertIn("txl.cuda.cta_sync()", rep.findings[0].evidence["hint"])
        self.assertEqual(self._run((bugs / "softmax_txl_bug_tail.py").read_text(), "tail", 200)[1], ["NOT_EQUIV"])
        self.assertEqual(self._run((bugs / "softmax_txl_bug_tail.py").read_text(), "tail", 128)[1], ["EQUIV"])  # hidden
        self.assertEqual(self._run((bugs / "softmax_txl_bug_nomax.py").read_text(), "nomax", 128)[1], ["EQUIV"])  # blind

    def test_glm_kernel_with_ex2_and_its_missing_log2e_mutant(self):
        src = (WORKSPACE / "results" / "pilot" / "20261115-205831" / "softmax_txl_r1.py").read_text()   # glm_txl_D3_r1
        rep, st = self._run(src, "glm_d3", 128)
        self.assertEqual(st, ["EQUIV"])
        self.assertEqual(rep.artifacts["runs"][0]["rewrites"]["candidate"]["ex2"], 1)
        bad = src.replace(" * txl.float32(LOG2E)", "")           # ex2(x - m): base-2 softmax, a real bug
        self.assertNotEqual(bad, src)
        self.assertEqual(self._run(bad, "glm_d3_nolog2e", 128)[1], ["NOT_EQUIV"])

    def test_candidate_with_less_smem_is_checked_alone(self):
        src = (WORKSPACE / "results" / "pilot" / "20261115-204306" / "softmax_txl_r2.py").read_text()   # glm_txl_D1_r2
        rep, st = self._run(src, "glm_d1", 128)
        run = rep.artifacts["runs"][0]
        self.assertEqual(st, ["EQUIV"])
        self.assertLess(run["dyn_smem"]["candidate"], run["dyn_smem"]["reference"])
        self.assertEqual(run["alone"]["status"], "CLEAN")

    def test_access_past_the_candidates_own_smem_is_out_of_bounds(self):
        # mutant of the reference: BLOCK 128 -> 64 halves the smem array, the hard-coded first tree step (s = 64)
        # then reads red[64..127]; within the reference's 512 bytes VOLTA's compare would only see UNINIT
        src = (self.tirx / "softmax_txl_ref.py").read_text().replace("BLOCK = 128", "BLOCK = 64")
        rep, st = self._run(src, "ref_block64", 128)
        self.assertEqual(st, ["OUT_OF_BOUNDS"])
        self.assertEqual(rep.artifacts["runs"][0]["alone"]["status"], "OUT_OF_BOUNDS")

    def test_trace_failure_is_a_compile_error(self):
        rep, st = self._run("import tirx_kernels.tirx_lite as txl\n\ndef make_kernel(rows, n):\n"
                            "    raise ValueError('no kernel for this n')\n", "broken", 128)
        self.assertEqual(st, ["BUILD_TRACE"])
        self.assertEqual([f.kind for f in rep.findings], ["compile_error"])
        self.assertFalse(rep.passed)


if __name__ == "__main__":
    unittest.main()
