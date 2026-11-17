"""Mutation benchmark (akc/evaluation/mutants.py): the generator and a regression guard for the classifier."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import BASES, FRESH, HELDOUT, SETS, classify_mutant, mutants, summarize  # noqa: E402
from akc.paths import ROOT  # noqa: E402
from akc.validation.changeaware.diff import diff_sources  # noqa: E402

OPT = ROOT / "workloads" / "softmax" / "cuda" / "softmax_opt.cu"


class MutantTests(unittest.TestCase):
    def test_operators_change_the_source_and_noops_are_cosmetic(self):
        src = OPT.read_text()
        ms = mutants(src, "cuda")
        self.assertTrue({"cmp", "idx1", "const", "define", "nosync", "aop", "math", "init", "negate", "dtype",
                         "rename", "format"} <= {m.op for m in ms})
        for m in ms:
            self.assertNotEqual(m.src, src, m.op)
            if not m.accept:                                         # rename / format: no semantic change
                self.assertEqual(diff_sources(src, m.src, "cuda"), [], m.desc)

    def test_edits_that_cannot_change_outputs_are_not_bugs(self):
        ms = mutants(OPT.read_text(), "cuda", per_op=20)
        larger = [m for m in ms if m.op == "idx1" and "declared" in m.desc]
        self.assertTrue(larger and not any(m.bug for m in larger))                  # red[BLOCK] -> red[BLOCK + 1]
        self.assertTrue(all(m.bug for m in ms if m.op == "idx1" and "declared" not in m.desc))
        self.assertFalse(any(m.bug for m in ms if m.op == "dtype"))                  # float -> double
        self.assertTrue(all(m.bug for m in ms if m.op in ("cmp", "nosync", "aop")))

    def test_tirx_mutants_stay_valid_python(self):
        import ast
        path = ROOT / "workloads" / "softmax" / "tirx" / "softmax_txl_ref.py"
        for m in mutants(path.read_text(), "tirx"):
            ast.parse(m.src)

    def test_fresh_set_loads(self):
        self.assertEqual([lab for lab, _, _ in FRESH], ["layernorm_txl_ref", "layernorm_txl_opt", "gemm_txl_ref",
                                                        "gemm_txl_opt"])
        self.assertEqual(SETS["all"], BASES + HELDOUT + FRESH)
        self.assertFalse({lab for lab, _, _ in FRESH} & {lab for lab, _, _ in BASES + HELDOUT})
        import ast
        for label, workload, rel in FRESH:
            path = ROOT / rel
            self.assertTrue(path.exists(), rel)
            ms = mutants(path.read_text(), "tirx", extended=True)
            self.assertGreaterEqual(len(ms), 15, label)
            for m in ms:
                ast.parse(m.src)
                self.assertNotEqual(m.src, path.read_text(), (label, m.op, m.line))
            self.assertIn("init", {m.op for m in ms}, label)                 # txl.float32(0.0) initial values

    def test_extended_operators_only_where_asked(self):
        opt = (ROOT / "workloads/gemm/tirx/gemm_txl_opt.py").read_text()
        ext = mutants(opt, "tirx", extended=True)
        self.assertEqual({m.desc for m in ext if m.op == "smem"}, {"shared size TILE -> TILE // 2"})
        self.assertFalse([m for m in mutants(opt, "tirx") if m.op in ("smem", "init")])
        ref = (ROOT / "workloads/softmax/tirx/softmax_txl_ref.py").read_text()   # a bases kernel: unchanged
        self.assertEqual([(m.op, m.line, m.desc) for m in mutants(ref, "tirx")],
                         [(m.op, m.line, m.desc) for m in mutants(ref, "tirx", extended=False)])

    def test_const_never_edits_a_float_exponent(self):
        src = (ROOT / "workloads/layernorm/tirx/layernorm_txl_opt.py").read_text()       # EPS = 1e-5
        self.assertFalse([m for m in mutants(src, "tirx", per_op=50) if m.op == "const" and "EPS" in
                          m.src.splitlines()[m.line - 1] and "1e-" in m.src.splitlines()[m.line - 1] and
                          m.src.splitlines()[m.line - 1] != src.splitlines()[m.line - 1]])

    def test_classifier_regression_guard(self):
        rows = []
        for label, workload, rel in BASES[:3] + HELDOUT[:1]:
            src = (ROOT / rel).read_text()
            lang = "tirx" if rel.endswith(".py") else "cuda"
            rows += [classify_mutant(workload, src, m, lang) for m in mutants(src, lang)]
        t = summarize(rows)["total"]
        self.assertEqual(t["none"], 0, t)                            # every semantic edit gets some category
        self.assertGreaterEqual(t["hit"] / t["semantic"], 0.95, t)
        self.assertEqual(t["spurious"], 0, t)


if __name__ == "__main__":
    unittest.main()
