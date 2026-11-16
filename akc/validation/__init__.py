"""Validators with one interface (interface.py) and their composition into configurations (pipeline.py).

    fixed_tests/   baseline: fixed test suite (configuration A)
    volta/         mechanism 1: VOLTA formal verification (Task 3)
    changeaware/   mechanism 2: change-aware testing (Task 2)
    runners/       test executors shared by fixed and change-aware tests (GPU harness, TIRx NumSim)
"""
from .interface import Candidate, Finding, ValidationContext, ValidationReport, Validator
from .pipeline import CONFIGS, PipelineResult, run_pipeline

__all__ = ["Candidate", "Finding", "ValidationContext", "ValidationReport", "Validator", "CONFIGS",
           "PipelineResult", "run_pipeline"]
