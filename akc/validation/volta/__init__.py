"""Validation mechanism 1: VOLTA formal verification (equivalence over the reals + races/deadlocks).

runner.py     run `volta compare` (and `volta analyze` for one kernel), parse its text output into a VoltaResult
layouts.py    per-workload memory layout / params / launch dims for VOLTA (CUDA and tirx-lite), manifest expectations
tirx.py       tirx-lite -> VOLTA: trace at a shape, rewrite TIRx's generated CUDA for VOLTA, compile to PTX
hints.py      actionable explanations of VOLTA verdicts and limitations
validator.py  VoltaValidator (the common Validator interface)
"""
from .validator import VoltaValidator

__all__ = ["VoltaValidator"]
