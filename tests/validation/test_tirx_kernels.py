"""Task 1: tirx-lite kernels for all three workloads (manifests, module contract, VOLTA layouts, VOLTA, GPU).

CPU tests run anywhere (login node). Tracing tests need tvm_tirx; VOLTA tests need nvcc + built VOLTA; GPU tests run
only where torch.cuda.is_available().
Run:  source env.sh && python -m unittest tests.validation.test_tirx_kernels -v
"""
from __future__ import annotations

import ast
import difflib
import shutil
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.backend.tirx import load_module, make_kernel, safety_scan  # noqa: E402
from akc.task.reference import OPS  # noqa: E402
from akc.task.testcase import TestCase  # noqa: E402
from akc.task.workload import load_workload  # noqa: E402
from akc.validation.volta.layouts import ROWS, expected_statuses, make_tirx_case, tirx_arrays  # noqa: E402
from akc.validation.volta.runner import VOLTA_BIN  # noqa: E402
from akc.validation.volta.tirx import TirxPtx, launch_info, parse_entry  # noqa: E402

# workload -> (reference, optimized candidate, {bug suffix: category}) as ported from workloads/<op>/cuda/bugs
EXPECTED = {
    "layernorm": ("layernorm_txl_ref", "layernorm_txl_opt",
                  {"nosync": "sync", "tail": "tiling", "offbyone": "indexing", "onepass": "numerical"}),
    "gemm": ("gemm_txl_ref", "gemm_txl_opt", {"nosync": "sync", "ktail": "tiling", "guard": "indexing"}),
}


def _have_tirx() -> bool:
    try:
        import tirx_kernels.tirx_lite  # noqa: F401
        import tvm_tirx  # noqa: F401
        return True
    except ImportError:
        return False


HAVE_TIRX = _have_tirx()
HAVE_VOLTA = shutil.which("nvcc") is not None and VOLTA_BIN.exists()


class ManifestTests(unittest.TestCase):
    def test_every_workload_has_a_tirx_contract_and_reference(self):
        for op in ("softmax", "layernorm", "gemm"):
            with self.subTest(op=op):
                wl, spec = load_workload(op), OPS[op]
                self.assertTrue(wl.has_reference("tirx"))
                self.assertEqual(wl.reference("tirx").lang, "tirx")
                self.assertEqual(wl.meta["tirx"]["factory_args"], wl.shape_keys)
                inputs = spec.make_inputs({k: 4 for k in wl.shape_keys}, "normal", 0)
                tensors = wl.meta["tirx"]["tensors"]
                self.assertEqual(tensors[:-1], list(inputs))          # inputs first (OpSpec names), output last
                self.assertNotIn(tensors[-1], inputs)

    def test_new_variants_exist_with_category_and_expectation(self):
        for op, (ref, opt, bugs) in EXPECTED.items():
            wl = load_workload(op)
            with self.subTest(op=op):
                self.assertEqual(wl.reference("tirx").name, ref)
                self.assertEqual([v.name for v in wl.candidates("tirx") if not v.bug], [opt])
                self.assertEqual(wl.variant(opt).expect_volta, "EQUIV")
                got = {v.name: v.bug["category"] for v in wl.bugs("tirx")}
                self.assertEqual(got, {f"{op}_txl_bug_{k}": c for k, c in bugs.items()})
                for v in wl.variants:
                    if v.lang == "tirx":
                        self.assertTrue(v.file.exists(), v.file)
                        self.assertEqual(safety_scan(v.file.read_text()), [], v.name)
                for k in bugs:                          # same category and VOLTA expectation as the CUDA twin
                    tv, cv = wl.variant(f"{op}_txl_bug_{k}"), wl.variant(f"{op}_bug_{k}")
                    self.assertEqual(tv.bug["category"], cv.bug["category"])
                    self.assertEqual(tv.expect_volta, cv.expect_volta)
                    self.assertEqual(tv.block, (256,) if op == "gemm" else (128,))

    def test_softmax_tirx_bugs_have_expectations(self):
        wl = load_workload("softmax")
        exp = {v.name: v.expect_volta for v in wl.bugs("tirx")}
        self.assertEqual(exp["softmax_txl_bug_nosync"], "RACE")
        self.assertEqual(exp["softmax_txl_bug_nomax"], "EQUIV")
        for k in ("tail", "offbyone"):
            e = exp[f"softmax_txl_bug_{k}"]
            self.assertEqual(expected_statuses(e, {"N": 256}), ("EQUIV",))
            self.assertEqual(expected_statuses(e, {"N": 200}), ("NOT_EQUIV",))

    def test_bugs_are_small_edits_of_the_optimized_kernel(self):
        # each bug module = the optimized module with its docstring replaced and 1-2 lines changed
        def body(src: str) -> list[str]:
            return src[src.index('"""', 3) + 3:].splitlines()          # without the module docstring

        for op, (_, opt, bugs) in EXPECTED.items():
            base = load_workload(op).variant(opt).file.read_text()
            for k in bugs:
                src = load_workload(op).variant(f"{op}_txl_bug_{k}").file.read_text()
                with self.subTest(bug=k):
                    ast.parse(src)
                    removed = [ln for ln in difflib.unified_diff(body(base), body(src), lineterm="", n=0)
                               if ln.startswith("-") and not ln.startswith("---")]
                    self.assertTrue(1 <= len(removed) <= 2, removed)


class VoltaLayoutTests(unittest.TestCase):
    """VOLTA memory layout for the new tensors (no tools needed)."""

    @staticmethod
    def _b(label, params, smem=0, grid=(2,), block=(128,)):
        return TirxPtx(label, Path(f"{label}.ptx"), Path(f"{label}.cu"), f"{label}_kernel", grid, block, smem, params)

    def test_layernorm_arrays(self):
        arrs = tirx_arrays("layernorm", {"N": 200})
        self.assertEqual(arrs, [("x", 4, ROWS * 200, "in"), ("w", 4, 200, "in"), ("b", 4, 200, "in"),
                                ("y", 4, ROWS * 200, "out")])

    def test_gemm_arrays(self):
        arrs = tirx_arrays("gemm", {"M": 12, "N": 12, "K": 40})
        self.assertEqual(arrs, [("A", 4, 12 * 40, "in"), ("B", 4, 40 * 12, "in"), ("C", 4, 12 * 12, "out")])

    def test_layernorm_case_uses_kernel_parameter_order(self):
        # TIRx orders tensor parameters by name (b, w, x, y); the case must follow the kernel, not the manifest
        order = ("b", "w", "x", "y")
        case = make_tirx_case("layernorm", self._b("ref", order, 512, block=(32,)), self._b("cand", order, 512),
                              {"N": 200})
        case.validate()
        self.assertEqual(case.params, tuple(f"ptr:{p}" for p in order))
        self.assertEqual(case.check_arrays, ("y",))
        self.assertEqual((case.ref_block, case.cand_block), ((32,), (128,)))

    def test_gemm_case_2d_grid_and_smem(self):
        case = make_tirx_case("gemm", self._b("ref", ("A", "B", "C"), 0, (2, 2), (256,)),
                              self._b("cand", ("A", "B", "C"), 2048, (2, 2), (256,)), {"M": 32, "N": 32, "K": 48})
        case.validate()
        self.assertEqual(case.grid, (2, 2))
        self.assertEqual(case.dyn_shared, 2048)
        self.assertEqual(case.check_arrays, ("C",))
        self.assertEqual([a.length for a in case.arrays], [32 * 48, 48 * 32, 32 * 32])


@unittest.skipUnless(HAVE_TIRX, "needs tvm_tirx")
class TraceTests(unittest.TestCase):
    """Every new tirx module traces at its VOLTA default shape; kernel parameters = the manifest tensors."""

    def test_trace_launch_and_parameters(self):
        for op in EXPECTED:
            wl = load_workload(op)
            shape = {**wl.volta["default"], **({"rows": ROWS} if "rows" in wl.shape_keys else {})}
            for v in [x for x in wl.variants if x.lang == "tirx"]:
                with self.subTest(kernel=v.name):
                    k = make_kernel(load_module(v.file, f"t_{v.name}"), wl.meta, shape)
                    grid, block, _ = launch_info(k)
                    self.assertEqual(block, v.block)
                    self.assertEqual(grid, (2,) if op == "layernorm" else (2, 2))
                    self.assertEqual(sorted(str(p.name) for p in k.func.params), sorted(wl.meta["tirx"]["tensors"]))
        src = make_kernel(load_module(wl.reference("tirx").file, "t_gemm_ref"), wl.meta, wl.volta["default"]).source()
        _, params = parse_entry(src)
        self.assertEqual(params, [("ptr", "A"), ("ptr", "B"), ("ptr", "C")])


@unittest.skipUnless(HAVE_TIRX and HAVE_VOLTA, "needs nvcc, built VOLTA and tvm_tirx")
class TirxVoltaTests(unittest.TestCase):
    """End to end through VoltaValidator at small shapes (a few seconds each)."""

    @classmethod
    def setUpClass(cls):
        from akc.validation.volta.validator import VoltaValidator
        cls.validator = VoltaValidator(timeout_s=120)

    def _status(self, op: str, name: str, shape: dict) -> str:
        from akc.validation.interface import Candidate, ValidationContext
        v = load_workload(op).variant(name)
        rep = self.validator.validate(Candidate(op, "tirx", v.file.read_text(), label=f"t_{name}"),
                                      ValidationContext(volta_shapes=[shape]))
        return rep.artifacts["runs"][0]["status"]

    def test_layernorm(self):
        self.assertEqual(self._status("layernorm", "layernorm_txl_opt", {"N": 100}), "EQUIV")
        self.assertEqual(self._status("layernorm", "layernorm_txl_bug_nosync", {"N": 100}), "RACE")
        self.assertEqual(self._status("layernorm", "layernorm_txl_bug_onepass", {"N": 100}), "EQUIV")   # blind spot

    def test_gemm(self):
        s = {"M": 12, "N": 12, "K": 40}
        self.assertEqual(self._status("gemm", "gemm_txl_opt", s), "EQUIV")
        self.assertEqual(self._status("gemm", "gemm_txl_bug_ktail", s), "NOT_EQUIV")
        self.assertIn(self._status("gemm", "gemm_txl_bug_guard", s), ("RACE", "OUT_OF_BOUNDS", "FOOTPRINT", "NOT_EQUIV"))


@unittest.skipUnless(torch.cuda.is_available() and HAVE_TIRX, "needs a GPU (run on a compute node) and tvm_tirx")
class TirxGpuTests(unittest.TestCase):
    """The tirx kernels on a real GPU through the shared runner (runners.gpu, lang "tirx")."""

    def _run(self, op, name, tests):
        from akc.validation.runners.gpu import run_tests_isolated
        v = load_workload(op).variant(name)
        return run_tests_isolated(op, "tirx", str(v.file), None, v.block, tests)

    def test_reference_and_opt_pass(self):
        shapes = {"layernorm": [{"rows": 64, "N": 256}, {"rows": 8, "N": 1000}],
                  "gemm": [{"M": 128, "N": 128, "K": 128}, {"M": 33, "N": 17, "K": 40}]}
        for op, (ref, opt, _) in EXPECTED.items():
            for name in (ref, opt):
                for r in self._run(op, name, [TestCase(op, s, repeat=2) for s in shapes[op]]):
                    self.assertTrue(r["passed"], f"{name} {r['shape']}: {r['message']}")

    def test_shape_and_numerical_bugs_caught(self):
        r = self._run("gemm", "gemm_txl_bug_ktail", [TestCase("gemm", {"M": 33, "N": 17, "K": 40})])[0]
        self.assertEqual(r["reason"], "mismatch", r["message"])
        r = self._run("layernorm", "layernorm_txl_bug_tail", [TestCase("layernorm", {"rows": 8, "N": 1000})])[0]
        self.assertFalse(r["passed"], r["message"])
        r = self._run("layernorm", "layernorm_txl_bug_onepass", [TestCase("layernorm", {"rows": 8, "N": 1000}, "offset")])[0]
        self.assertFalse(r["passed"], r["message"])


if __name__ == "__main__":
    unittest.main()
