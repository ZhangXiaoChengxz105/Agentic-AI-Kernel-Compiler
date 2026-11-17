"""D2: the compiler-checks gate (akc/validation/compiler_checks/) and configurations A0 / A / B / C.

  SanitizerParserTests   the compute-sanitizer log parser on saved logs (tests/data/sanitizer/, trimmed copies of
                         logs/sanitizer_trial/; plus every saved trial log when logs/ is present)
  FindingMappingTests    report heads -> Finding kinds; sanitizer / NumSim / static results -> Findings; feedback text
  PipelineTests          configurations, order and stop rules with fake validators; A0 = the old A; the report cache
  TirxStaticTests        static checks pass on the tirx references, opts and injected bugs, fail on an IR-contract
                         violation (with and without check_ir) at the right line (needs tvm_tirx; ~1 min)
  GpuGateTests           GPU only: nosync -> race, gemm guard -> out_of_bounds, references clean
Run: source env.sh && python -m unittest tests.validation.test_compiler_checks -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.feedback.render import render  # noqa: E402
from akc.paths import BUILD, LOGS, RESULTS, ROOT  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation import pipeline  # noqa: E402
from akc.validation.compiler_checks import sanitizer as san  # noqa: E402
from akc.validation.compiler_checks.tirx_static import gate_shapes, numsim_shape  # noqa: E402
from akc.validation.compiler_checks.validator import CompilerChecksValidator  # noqa: E402
from akc.validation.fixed_tests.validator import FixedTestsValidator  # noqa: E402
from akc.validation.interface import Candidate, Finding, ValidationContext, ValidationReport  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "sanitizer"
CPU = ValidationContext(device="cpu")


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def _gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


HAVE_TIRX = _has("tvm_tirx") and _has("tirx_kernels.tirx_lite")


class SanitizerParserTests(unittest.TestCase):
    def _p(self, name, entry, tool):
        return san.parse_sanitizer((DATA / name).read_text(), entry, tool)

    def test_memcheck_out_of_bounds_with_line(self):
        r = self._p("gemm_bug_guard.memcheck.log", "gemm_bug_guard", "memcheck")
        self.assertTrue(r["caught"])
        self.assertEqual({x["kind"] for x in r["reports"]}, {"out_of_bounds"})
        self.assertEqual(r["reports"][0]["locs"][0], ("gemm_bug_guard.cu", 36))
        self.assertEqual(r["error_summary"], 35)

    def test_racecheck_race_with_hazard_count(self):
        r = self._p("softmax_bug_nosync.racecheck.log", "softmax_bug_nosync", "racecheck")
        self.assertTrue(r["caught"])
        self.assertEqual({x["kind"] for x in r["reports"]}, {"race"})
        self.assertEqual(r["by_severity"], {"error": 1, "warning": 2, "info": 0})
        self.assertEqual(r["reports"][0]["hazards"], 8192 + 4096 + 432)
        self.assertEqual(r["racecheck_summary"], [18, 4, 14])

    def test_initcheck_uninitialized_shared(self):
        r = self._p("softmax_opt_const_L23.initcheck.log", "softmax_opt", "initcheck")
        self.assertTrue(r["caught"])
        self.assertEqual({x["kind"] for x in r["reports"]}, {"uninitialized"})
        self.assertIn(("softmax_opt_const_L23.cu", 24), r["reports"][0]["locs"])

    def test_clean_logs_and_other_kernels_are_not_caught(self):
        for name, tool in (("softmax_opt.synccheck.log", "synccheck"), ("softmax_opt.racecheck.log", "racecheck")):
            r = self._p(name, "softmax_opt", tool)
            self.assertFalse(r["caught"], name)
            self.assertEqual(r["n_reports"], 0)
        r = self._p("gemm_bug_guard.memcheck.log", "some_other_kernel", "memcheck")   # filtered by entry name
        self.assertFalse(r["caught"])
        self.assertEqual(r["not_ours"], 2)

    @unittest.skipUnless((LOGS / "sanitizer_trial").is_dir() and (RESULTS / "eval" / "sanitizer_trial.json").exists(),
                         "needs the saved trial logs (logs/sanitizer_trial)")
    def test_same_verdicts_as_the_trial_on_every_saved_log(self):
        d = json.loads((RESULTS / "eval" / "sanitizer_trial.json").read_text())
        entry = {k["label"]: k.get("entry", "") for k in d["kernels"]}
        n = 0
        for r in d["runs"]:
            if r["tool"] not in san.TOOLS or not (ROOT / r["log"]).exists():
                continue
            got = san.parse_sanitizer((ROOT / r["log"]).read_text(), entry[r["label"]], r["tool"])
            self.assertEqual(got["caught"], r["result"]["caught"], r["log"])
            n += 1
        self.assertGreater(n, 100)


@unittest.skipUnless(san.shutil.which("nvcc"), "needs nvcc (+ nvdisasm) to rebuild the cubin")
class TirxRacecheckLocationTests(unittest.TestCase):
    """A real racecheck log of softmax_txl_bug_nosync (job 24061664) and its generated CUDA: TIRx's shared accesses sit
    in inlined helpers, so the reports name tvm_builtin_ptx_*_shared_f32, not the kernel. They must still count as
    ours (by file name), and the cubin's inline line table must move them to the call sites in the kernel body."""

    def test_helper_reports_are_ours_and_map_to_kernel_lines(self):
        with tempfile.TemporaryDirectory() as t:
            cu = Path(t) / "softmax_txl_bug_nosync_rows4_N129.cu"
            cu.write_text((DATA / cu.name).read_text())
            cubin, _ = san.build_cubin(cu, Path(t))
            sass = san.sass_line_map(cubin, "softmax_txl_kernel")
            self.assertEqual((sass.get(0x260), sass.get(0x210)), (116, 112))     # STS / LDS of two reduction steps
            text = (DATA / "softmax_txl_bug_nosync.racecheck.log").read_text()
            self.assertFalse(san.parse_sanitizer(text, "softmax_txl_kernel", "racecheck")["caught"])  # by name only
            parsed = san.parse_sanitizer(text, "softmax_txl_kernel", "racecheck", files={cu.name})
            self.assertTrue(parsed["caught"])
            files = {cu.name: {"cuda": str(cu), "shape": {"rows": 4, "N": 129}, "sass": sass}}
            f = CompilerChecksValidator._tool_findings("racecheck", parsed, files, {"log": "x"})[0]
        self.assertEqual(f.kind, "race")
        self.assertEqual(f.location, "generated CUDA line 116 (write) and line 112 (read)")
        self.assertIn("v[((int)threadIdx.x)]", f.evidence["cuda"]["code"])
        self.assertTrue(f.evidence["cuda"]["in_kernel"])


class FindingMappingTests(unittest.TestCase):
    def test_report_kind_per_tool(self):
        self.assertEqual(san.report_kind("memcheck", "Invalid __global__ write of size 4 bytes"), "out_of_bounds")
        self.assertEqual(san.report_kind("memcheck", "Invalid __shared__ read of size 4 bytes"), "out_of_bounds")
        self.assertEqual(san.report_kind("memcheck", "Misaligned Shared or Local Address"), "out_of_bounds")
        self.assertEqual(san.report_kind("racecheck", "Error: Race reported between Write access at k+0x10"), "race")
        self.assertEqual(san.report_kind("initcheck", "Uninitialized __global__ memory read of size 4 bytes"),
                         "uninitialized")
        self.assertIsNone(san.report_kind("initcheck", "Unused memory in allocation 0x7f of size 64 bytes"))
        self.assertEqual(san.report_kind("synccheck", "Barrier error detected. Divergent thread(s) in warp"), "deadlock")

    def _cu(self, tmp: Path) -> Path:
        cu = tmp / "cand_rows4_N129.cu"
        cu.write_text("\n".join([
            "__forceinline__ __device__ void tvm_builtin_ptx_st_shared_f32(uint32_t __addr, float __value) {",
            '  asm volatile("st.shared.f32 [%0], %1;" :  : "r"(__addr), "f"(__value) : "memory");',
            "}",
            'extern "C" __global__ void __launch_bounds__(128) k(float* __restrict__ x_ptr, float* __restrict__ y_ptr);',
            'extern "C" __global__ void __launch_bounds__(128) k(float* __restrict__ x_ptr, float* __restrict__ y_ptr) {',
            "  extern __shared__ __align__(64) uchar v_ptr[];",
            "  tvm_builtin_ptx_st_shared_f32(0, x_ptr[threadIdx.x]);",
            "}"]) + "\n")
        return cu

    def test_sanitizer_reports_become_findings_at_the_kernel_line(self):
        with tempfile.TemporaryDirectory() as t:
            cu = self._cu(Path(t))
            files = {cu.name: {"cuda": str(cu), "shape": {"rows": 4, "N": 129}}}
            parsed = {"reports": [
                {"severity": "error", "head": "Error: Race reported between Write access at k+0x10 in cand_rows4_N129.cu:2",
                 "kind": "race", "locs": [("cand_rows4_N129.cu", 2), ("cand_rows4_N129.cu", 7)], "hazards": 12},
                {"severity": "info", "head": "Info: Race reported ...", "kind": "race", "locs": [], "hazards": 1}]}
            fs = CompilerChecksValidator._tool_findings("racecheck", parsed, files, {"log": "x.log"})
        err = [f for f in fs if f.severity == "error"]
        self.assertEqual(len(err), 1)
        self.assertEqual(err[0].kind, "race")
        self.assertEqual(err[0].location, "generated CUDA line 7")           # the kernel body, not the helper
        self.assertTrue(err[0].evidence["cuda"]["in_kernel"])
        self.assertEqual([f.kind for f in fs if f.severity == "info"], ["race"])
        text = render([ValidationReport.build("compiler_checks", fs, True, 0.0)])
        self.assertIn("in the CUDA that TIRx generated from your module (no Python line is known), line 7", text)
        self.assertIn("txl.cuda.cta_sync()", text)                         # the race hint

    def test_helper_only_location_names_the_helper(self):
        with tempfile.TemporaryDirectory() as t:
            cu = self._cu(Path(t))
            c = CompilerChecksValidator._tool_findings("memcheck", {"reports": [
                {"severity": "error", "head": "Invalid __shared__ write of size 4 bytes", "kind": "out_of_bounds",
                 "locs": [("cand_rows4_N129.cu", 2)], "hazards": None}]},
                {cu.name: {"cuda": str(cu), "shape": {"rows": 4, "N": 129}}}, {"log": "x"})[0]
        self.assertEqual(c.kind, "out_of_bounds")
        self.assertIn("tvm_builtin_ptx_st_shared_f32", c.location)

    def test_numsim_statuses(self):
        ns = {"shape": {"rows": 2, "N": 129},
              "racecheck": {"verdict": "error", "seconds": 0.5, "findings": [
                  {"status": "error", "kind": "data_race", "message": "unordered read/write", "summary": "read/write",
                   "lines": [38, 36], "callers": [48]}]},
              "synccheck": {"verdict": "review", "seconds": 0.2, "findings": [
                  {"status": "review", "kind": "barrier_divergence", "message": "m", "lines": [40], "callers": []},
                  {"status": "incomplete", "kind": "analysis_incomplete", "message": "m", "lines": [], "callers": []}]}}
        fs, decided = CompilerChecksValidator()._numsim_findings(ns, Path("cand.py"), [], {})
        got = {(f.kind, f.severity) for f in fs}
        self.assertEqual(got, {("race", "error"), ("deadlock", "warning"), ("unsupported", "info")})
        race = next(f for f in fs if f.kind == "race")
        self.assertEqual(race.location, "lines 38 and 36 (in a helper called from line 48)")
        self.assertIn("of your kernel module", render([ValidationReport.build("compiler_checks", fs, True, 0.0)]))
        self.assertTrue(decided)
        crash = {"shape": {}, "racecheck": {"verdict": "crash", "error": "NumSim: out-of-bounds store", "lines": [12]},
                 "synccheck": {"verdict": "incomplete", "findings": [
                     {"status": "incomplete", "kind": "x", "message": "m", "lines": [], "callers": []}]}}
        fs, decided = CompilerChecksValidator()._numsim_findings(crash, Path("cand.py"), [], {})
        self.assertIn(("out_of_bounds", "error"), {(f.kind, f.severity) for f in fs})
        self.assertFalse(decided)                                          # synccheck incomplete: no verdict

    def test_static_errors_are_compile_errors_with_the_line(self):
        st = {"errors": [{"stage": "ir_contract", "shape": {"rows": 4, "N": 129}, "lines": [44],
                          "message": "low-level IR contract failed with 1 violation(s)\n- root: buffer_load (TensorLoad, scope=global)"},
                         {"stage": "ir_contract", "shape": {"rows": 4, "N": 1000}, "lines": [44],
                          "message": "low-level IR contract failed with 1 violation(s)\n- root: buffer_load (TensorLoad, scope=global)"}]}
        fs = CompilerChecksValidator._static_findings(st, Path("cand.py"))
        self.assertEqual(len(fs), 1)                                       # same error at two shapes: one finding
        self.assertEqual((fs[0].kind, fs[0].severity, fs[0].location), ("compile_error", "error", "line 44"))
        self.assertIn("buffer_load (TensorLoad, scope=global)", fs[0].message)
        self.assertEqual(len(fs[0].evidence["shapes"]), 2)

    def test_legacy_cuda_is_info_only(self):
        wl = load_workload("softmax")
        v = wl.variant("softmax_opt")
        r = CompilerChecksValidator().validate(Candidate.from_file("softmax", v.file, entry=v.name), CPU)
        self.assertTrue(r.passed)
        self.assertFalse(r.decided)
        self.assertEqual([(f.kind, f.severity) for f in r.findings], [("skipped", "info")])
        self.assertIn("legacy CUDA", r.findings[0].message)

    def test_gate_shapes(self):
        sm = (ROOT / "workloads/softmax/tirx/softmax_txl_ref.py").read_text()
        self.assertEqual(gate_shapes("softmax", sm), [{"rows": 4, "N": 129}, {"rows": 4, "N": 1000}])
        glm = (ROOT / "results/pilot/20261115-204306/softmax_txl_r2.py").read_text()     # size regime at N = 12284
        self.assertIn({"rows": 2, "N": 12285}, gate_shapes("softmax", glm))
        gm = (ROOT / "workloads/gemm/tirx/gemm_txl_opt.py").read_text()
        self.assertEqual(gate_shapes("gemm", gm), [{"M": 33, "N": 33, "K": 40}, {"M": 64, "N": 64, "K": 64}])
        self.assertEqual(numsim_shape("softmax", sm), {"rows": 2, "N": 129})
        self.assertEqual(numsim_shape("gemm", gm), {"M": 12, "N": 12, "K": 20})
        for w, src in (("softmax", glm), ("gemm", gm)):
            wl = load_workload(w)
            self.assertTrue(all(wl.in_domain(s) for s in gate_shapes(w, src)))


class _Fake:
    """A validator with a scripted result; records the contexts it was called with."""
    calls: list = []

    def __init__(self, name, findings=()):
        self.name, self.findings = name, list(findings)

    def validate(self, cand, ctx):
        _Fake.calls.append((self.name, dict(ctx.extra), ctx.volta_shapes))
        art = {"suggested_volta_shapes": [{"N": 200}]} if self.name == "changeaware" else {}
        return ValidationReport.build(self.name, [Finding(**f) for f in self.findings], True, 0.0, **art)


def _fakes(**findings):
    def make(name):
        return lambda: _Fake(name, [dict(kind=k, severity="error", validator=name, message=k) for k in findings.get(name, [])])
    return {n: make(n) for n in ("compiler_checks", "fixed_tests", "changeaware", "volta")}


class PipelineTests(unittest.TestCase):
    cand = Candidate("softmax", "tirx", "x = 1\n", label="fake")

    def setUp(self):
        _Fake.calls = []

    def _run(self, config, findings=None, run_all=False, **kw):
        with mock.patch.dict(pipeline.VALIDATORS, _fakes(**(findings or {}))):
            return pipeline.run_pipeline(self.cand, config, CPU.with_(extra={"run_all": run_all}), **kw)

    def test_configurations(self):
        self.assertEqual(pipeline.CONFIGS, {
            "A0": ("fixed_tests",), "A": ("compiler_checks", "fixed_tests"),
            "B": ("compiler_checks", "fixed_tests", "volta"),
            "C": ("compiler_checks", "fixed_tests", "changeaware", "volta")})
        with self.assertRaises(ValueError):
            pipeline.run_pipeline(self.cand, "D", CPU)

    def test_order_and_volta_shapes(self):
        res = self._run("C")
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks", "fixed_tests", "changeaware", "volta"])
        self.assertEqual(_Fake.calls[-1][2], [{"N": 200}])                 # C: change-aware shapes for VOLTA
        self._run("B")
        self.assertEqual(_Fake.calls[-1][0], "volta")
        self.assertIsNone(_Fake.calls[-1][2])                              # B: the workload default shape
        self.assertTrue(res.passed)
        self.assertIsNone(res.stopped_after)

    def test_static_error_stops_even_with_run_all(self):
        res = self._run("C", {"compiler_checks": ["compile_error"]}, run_all=True)
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks"])
        self.assertEqual(res.stopped_after, "compiler_checks")
        self.assertEqual(res.skipped, ["fixed_tests", "changeaware", "volta"])
        self.assertFalse(res.passed)

    def test_sanitizer_error_stops_by_default_but_not_with_run_all(self):
        res = self._run("C", {"compiler_checks": ["race"]})
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks"])
        self.assertFalse(res.passed)
        res = self._run("C", {"compiler_checks": ["race"]}, run_all=True)
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks", "fixed_tests", "changeaware", "volta"])
        self.assertFalse(res.passed)
        self.assertEqual(res.per_validator["compiler_checks"]["kinds"], ["race"])
        self.assertFalse(res.per_validator["volta"]["caught"])

    def test_test_failure_stops_by_default(self):
        res = self._run("C", {"fixed_tests": ["mismatch"]})
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks", "fixed_tests"])

    def test_gate_turns_off_numsim_racecheck_in_the_test_runners(self):
        self._run("C")
        extra = {name: ex for name, ex, _ in _Fake.calls}
        self.assertNotIn("numsim_racecheck", extra["compiler_checks"])
        self.assertIs(extra["fixed_tests"]["numsim_racecheck"], False)
        self.assertIs(extra["changeaware"]["numsim_racecheck"], False)
        _Fake.calls = []
        self._run("A0")
        self.assertNotIn("numsim_racecheck", _Fake.calls[0][1])           # A0 = the old A: Racecheck stays on

    def test_cache_runs_each_validator_once(self):
        for device, fixed_runs in (("cpu", 2), ("gpu", 1)):
            _Fake.calls, cache = [], {}
            with mock.patch.dict(pipeline.VALIDATORS, _fakes()):
                for cfg in ("A0", "A", "B", "C"):
                    res = pipeline.run_pipeline(self.cand, cfg, ValidationContext(device=device, extra={"run_all": True}),
                                                cache=cache)
            names = [n for n, _, _ in _Fake.calls]
            # CPU: A0's NumSim tests run Racecheck, A/B/C's do not (the gate ran it), so fixed_tests runs twice; on the
            # GPU once. volta twice: B at the default shape, C at the change-aware shapes
            self.assertEqual(sorted(names), sorted(["compiler_checks", "changeaware", "volta", "volta"]
                                                   + ["fixed_tests"] * fixed_runs), device)
            self.assertEqual(res.cached, ["compiler_checks", "fixed_tests"])     # C reuses A's and B's reports

    def test_skip(self):
        res = self._run("C", skip=("volta",))
        self.assertEqual([r.validator for r in res.reports], ["compiler_checks", "fixed_tests", "changeaware"])
        self.assertEqual(res.skipped, ["volta"])

    def test_A0_equals_the_old_A(self):
        # the old configuration A was exactly ("fixed_tests",) with the plain context: same validator, same result
        wl = load_workload("softmax")
        v = wl.variant("softmax_bug_tail")
        cand = Candidate.from_file("softmax", v.file, entry=v.name, label=v.name)
        new = pipeline.run_pipeline(cand, "A0", CPU)
        old = FixedTestsValidator().validate(cand, CPU.with_(config="A"))
        self.assertEqual([r.validator for r in new.reports], ["fixed_tests"])
        strip = lambda r: (r.passed, r.decided, [(f.kind, f.severity, f.message) for f in r.findings],  # noqa: E731
                           r.artifacts.get("n_tests"))
        self.assertEqual(strip(new.reports[0]), strip(old))


@unittest.skipUnless(HAVE_TIRX, "needs tvm_tirx / tirx_kernels")
class TirxStaticTests(unittest.TestCase):
    """One small shape per kernel, 4 child processes at a time (each imports tvm: ~2-5 s, compile ~2 s)."""

    @classmethod
    def setUpClass(cls):
        from akc.validation.compiler_checks.tirx_static import run_static
        cls.tmp = Path(tempfile.mkdtemp(dir=BUILD / "tmp" if (BUILD / "tmp").is_dir() else None))
        src = (ROOT / "workloads/softmax/tirx/softmax_txl_ref.py").read_text()
        bad = src.replace("txl.ptx.ld.global_.f32(v, x.ptr_to([r, col]))\n                txl.assign(m, txl.max(m, v))",
                          "txl.assign(m, txl.max(m, x[r, col]))", 1)
        assert bad != src
        (cls.tmp / "bad.py").write_text(bad)
        (cls.tmp / "bad_nocheck.py").write_text(bad.replace("@txl.kernel()", "@txl.kernel(check_ir=False)"))
        cls.bad_line = next(i + 1 for i, ln in enumerate(bad.splitlines()) if "x[r, col]" in ln)
        jobs = []
        for w in ("softmax", "layernorm", "gemm"):
            wl = load_workload(w)
            shape = {"M": 33, "N": 33, "K": 40} if w == "gemm" else {"rows": 4, "N": 129}
            for v in wl.variants:
                if v.lang == "tirx":
                    jobs.append((v.name, w, str(v.file), shape))
        jobs += [("bad", "softmax", str(cls.tmp / "bad.py"), {"rows": 4, "N": 129}),
                 ("bad_nocheck", "softmax", str(cls.tmp / "bad_nocheck.py"), {"rows": 4, "N": 129})]
        with ThreadPoolExecutor(4) as ex:
            res = ex.map(lambda j: (j[0], run_static(j[1], j[2], [j[3]], cls.tmp / "cuda", j[0])), jobs)
            cls.res = dict(res)

    def test_references_opts_and_injected_bugs_pass(self):
        names = [n for n in self.res if not n.startswith("bad")]
        self.assertEqual(len(names), 16)                                    # 5 refs / opts + 11 bugs
        for n in names:
            r = self.res[n]
            self.assertEqual(r["errors"], [], n)
            self.assertIsNone(r["error"], n)
            s = r["shapes"][0]
            self.assertEqual(set(s["stages"]), {"trace", "ir_contract", "well_formed", "verify_memory", "compile"}, n)
            self.assertTrue(Path(s["cuda"]).exists(), n)
            self.assertTrue(s["entry"] and s["params"] and s["grid"] and s["block"], n)

    def test_contract_violation_fails_at_its_line(self):
        for n in ("bad", "bad_nocheck"):                 # traced with check_ir, and with check_ir=False
            errs = self.res[n]["errors"]
            self.assertEqual([e["stage"] for e in errs], ["ir_contract"], n)
            self.assertIn("buffer_load", errs[0]["message"])
            self.assertEqual(errs[0]["lines"], [self.bad_line], n)
        self.assertEqual(self.res["bad"]["shapes"][0]["stages"]["trace"]["status"], "error")
        self.assertEqual(self.res["bad_nocheck"]["shapes"][0]["stages"]["trace"]["status"], "ok")

    def test_gemm_parameter_order(self):
        s = self.res["gemm_txl_opt"]["shapes"][0]
        self.assertEqual(s["params"], ["A", "B", "C"])
        self.assertEqual((s["grid"], s["block"]), ([3, 3], [256]))


@unittest.skipUnless(_gpu() and HAVE_TIRX and san.find_compute_sanitizer(), "GPU + compute-sanitizer only")
class GpuGateTests(unittest.TestCase):
    GPU = ValidationContext(device="gpu")

    def _gate(self, workload, name):
        v = load_workload(workload).variant(name)
        return CompilerChecksValidator().validate(Candidate.from_file(workload, v.file, label=v.name), self.GPU)

    def test_nosync_is_a_race(self):
        r = self._gate("softmax", "softmax_txl_bug_nosync")
        self.assertFalse(r.passed)
        self.assertIn("race", {f.kind for f in r.findings if f.severity == "error"})
        self.assertTrue(r.artifacts["sanitizer"]["racecheck"]["caught"])

    def test_gemm_guard_is_out_of_bounds(self):
        r = self._gate("gemm", "gemm_txl_bug_guard")
        self.assertFalse(r.passed)
        self.assertIn("out_of_bounds", {f.kind for f in r.findings if f.severity == "error"})
        self.assertTrue(r.artifacts["sanitizer"]["memcheck"]["caught"])

    def test_references_are_clean(self):
        for w, n in (("softmax", "softmax_txl_ref"), ("gemm", "gemm_txl_opt"), ("layernorm", "layernorm_txl_opt")):
            r = self._gate(w, n)
            self.assertTrue(r.passed, [f.message for f in r.findings])
            self.assertTrue(r.decided, n)
            self.assertTrue(all(v.get("passed") for v in r.artifacts["value_check"]), r.artifacts["value_check"])


if __name__ == "__main__":
    unittest.main()
