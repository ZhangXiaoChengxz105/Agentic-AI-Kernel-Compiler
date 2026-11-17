"""Mutation benchmark (akc/evaluation/mutants.py): the generator and a regression guard for the classifier."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.mutants import BASES, HELDOUT, classify_mutant, mutants, summarize  # noqa: E402
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
