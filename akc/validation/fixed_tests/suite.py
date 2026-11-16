"""Configuration A's fixed test suite: the same small, typical test set for every kernel revision.

Taken from the workload manifest ("fixed_suite"): power-of-two shapes (divisible by every tile/block size
we use) with well-behaved N(0,1) data and 3 repeats. Realistic on purpose (what people usually test); by
construction it misses tail/edge bugs and numerical-range bugs, which is the gap change-aware tests fill.
"""
from __future__ import annotations

from ...task.testcase import TestCase
from ...task.workload import load_workload


def fixed_suite(workload: str, repeat: int = 3) -> list[TestCase]:
    wl = load_workload(workload)
    return [TestCase(workload, dict(shape), "normal", seed=i, repeat=repeat, tag="fixed")
            for i, shape in enumerate(wl.meta["fixed_suite"])]
