"""akc eval-bugs: which correct version the injected bugs are compared against.
Run:  source env.sh && python -m unittest tests.evaluation.test_bugset -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest import mock  # noqa: E402

from akc.evaluation.bugset import evaluate, parent_variant  # noqa: E402
from akc.task.workload import list_workloads, load_workload  # noqa: E402
from akc.validation import pipeline  # noqa: E402
from akc.validation.interface import Candidate, Finding, ValidationContext, ValidationReport  # noqa: E402


class ParentVariantTests(unittest.TestCase):
    def test_cuda_uses_the_optimized_kernel(self):
        for w in list_workloads():
            self.assertEqual(parent_variant(load_workload(w), "cuda").name, f"{w}_opt")

    def test_tirx_falls_back_to_the_reference(self):
        # GPU run 24036388 crashed here (StopIteration): tirx bugs are edits of the reference, no "opt" exists
        wl = load_workload("softmax")
        self.assertTrue(wl.bugs("tirx"))
        self.assertEqual(parent_variant(wl, "tirx").name, wl.meta["references"]["tirx"])


class EvaluateTests(unittest.TestCase):
    def test_every_config_through_the_pipeline_with_per_validator_numbers(self):
        calls = []

        class Fake:
            def __init__(self, name, kinds=()):
                self.name, self.kinds = name, kinds

            def validate(self, cand, ctx):
                calls.append((self.name, ctx.extra.get("run_all")))
                f = [Finding(k, "error", self.name, k) for k in self.kinds]
                art = {"suggested_volta_shapes": [{"N": 200}]} if self.name == "changeaware" else {}
                return ValidationReport(self.name, not f, True, f, 1.5, art)
        fakes = {"compiler_checks": lambda: Fake("compiler_checks", ["race"]), "fixed_tests": lambda: Fake("fixed_tests"),
                 "changeaware": lambda: Fake("changeaware"), "volta": lambda: Fake("volta", ["race"])}
        with mock.patch.dict(pipeline.VALIDATORS, fakes):
            res = evaluate(Candidate("softmax", "tirx", "x = 1\n", label="fake"), ValidationContext(device="gpu"))
        self.assertEqual({c: r["caught"] for c, r in res.items()}, {"A0": False, "A": True, "B": True, "C": True})
        self.assertTrue(all(ra for _, ra in calls))                     # run_all: the gate's race does not stop
        self.assertEqual(res["C"]["per_validator"]["compiler_checks"], {"caught": True, "seconds": 1.5, "decided": True,
                                                                        "kinds": ["race"], "cached": True})
        self.assertEqual(list(res["C"]["per_validator"]), ["compiler_checks", "fixed_tests", "changeaware", "volta"])
        self.assertEqual(res["C"]["seconds"], 6.0)                      # own seconds of the 4 validators, cached or not
        self.assertEqual(len(calls), 5)                                 # gate, fixed, volta(B), changeaware, volta(C)


if __name__ == "__main__":
    unittest.main()
