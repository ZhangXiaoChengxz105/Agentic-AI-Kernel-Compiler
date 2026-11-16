"""akc eval-bugs: which correct version the injected bugs are compared against.
Run:  source env.sh && python -m unittest tests.evaluation.test_bugset -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from akc.evaluation.bugset import parent_variant  # noqa: E402
from akc.task.workload import list_workloads, load_workload  # noqa: E402


class ParentVariantTests(unittest.TestCase):
    def test_cuda_uses_the_optimized_kernel(self):
        for w in list_workloads():
            self.assertEqual(parent_variant(load_workload(w), "cuda").name, f"{w}_opt")

    def test_tirx_falls_back_to_the_reference(self):
        # GPU run 24036388 crashed here (StopIteration): tirx bugs are edits of the reference, no "opt" exists
        wl = load_workload("softmax")
        self.assertTrue(wl.bugs("tirx"))
        self.assertEqual(parent_variant(wl, "tirx").name, wl.meta["references"]["tirx"])


if __name__ == "__main__":
    unittest.main()
