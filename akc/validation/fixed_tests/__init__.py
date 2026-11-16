"""Baseline: the fixed test suite (configuration A)."""
from .suite import fixed_suite
from .validator import FixedTestsValidator

__all__ = ["FixedTestsValidator", "fixed_suite"]
