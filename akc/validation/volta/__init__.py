"""Validation mechanism 1: VOLTA formal verification (equivalence over the reals + races/deadlocks).

runner.py     run `volta compare`, parse its text output into a VoltaResult (12 normalized statuses)
layouts.py    per-workload memory layout / params / launch dims for VOLTA, manifest expectations
hints.py      actionable explanations of VOLTA verdicts and limitations
validator.py  VoltaValidator (the common Validator interface)
"""
from .validator import VoltaValidator

__all__ = ["VoltaValidator"]
