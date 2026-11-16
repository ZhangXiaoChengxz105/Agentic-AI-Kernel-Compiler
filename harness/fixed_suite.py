"""Configuration A's "fixed tests": a small, typical test set, the same for every kernel revision.

Deliberately realistic, i.e. what people usually test: power-of-two sizes (divisible by every tile
and block size we use) and well-behaved N(0,1) data. By construction it misses tail/edge bugs and
numerical-range bugs; that is the gap change-aware tests (Task 2) are meant to fill.
"""
from __future__ import annotations

from .testcase import TestCase

FIXED: dict[str, list[TestCase]] = {
    "softmax": [TestCase("softmax", {"rows": 64, "N": n}, "normal", seed=i, tag="fixed")
                for i, n in enumerate((256, 1024, 4096))],
    "layernorm": [TestCase("layernorm", {"rows": 64, "N": n}, "normal", seed=i, tag="fixed")
                  for i, n in enumerate((256, 1024, 4096))],
    "gemm": [TestCase("gemm", {"M": n, "N": n, "K": n}, "normal", seed=i, tag="fixed")
             for i, n in enumerate((128, 256, 512))],
}
