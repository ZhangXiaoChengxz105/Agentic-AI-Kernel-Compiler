"""The one interface every validator implements, so the agent loop, feedback and evaluation only
need to understand a single result format.

    Validator.validate(candidate: Candidate, ctx: ValidationContext) -> ValidationReport

Candidate          the kernel under test (+ the version it should be compared against, for change-aware tests)
ValidationContext  how to validate this time (configuration, tests, VOLTA shapes, device, budget)
Finding            one problem (or limitation) found, with evidence the model can act on
ValidationReport   everything one validator found + `decided` (for verification coverage) + `seconds`
                   (for validation overhead)

Severity
  error    a definite bug (wrong output, race, out-of-bounds, compile error, ...): candidate fails
  warning  suspicious or harness-side problem that does not prove a bug
  info     the validator could not decide (VOLTA unsupported/timeout, test not executable here)

Parent policy (change-aware tests): `parent` must be the last candidate that PASSED validation (not simply the
previous revision), so changes made in rejected revisions stay in the diff; for the first candidate the
workload's reference kernel is used. (Pilot evidence: a bug introduced in round 2 survived a round-3 edit that
only touched an initializer.)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol

from ..paths import BUILD

SEVERITIES = ("error", "warning", "info")
FINDING_KINDS = (
    "compile_error", "runtime_error", "race", "deadlock", "not_equivalent", "mismatch", "nonfinite",
    "out_of_bounds", "unwritten", "uninitialized", "footprint", "div_by_zero", "nondeterministic",
    "unsupported", "timeout", "skipped", "harness_error",
)


@dataclass
class Candidate:
    workload: str                          # e.g. "softmax"
    lang: str                              # "cuda" | "tirx"
    source: str                            # full kernel source text
    entry: str | None = None               # CUDA extern "C" entry name (None: the single entry in the PTX)
    block: tuple[int, ...] | None = None   # launch block (None: the workload's default)
    parent: "Candidate | None" = None      # last ACCEPTED version (see module docstring)
    label: str = ""                        # human-readable name for reports
    path: Path | None = None               # file holding `source` (created on demand)

    @staticmethod
    def from_file(workload: str, path: str | Path, lang: str | None = None, **kw) -> "Candidate":
        path = Path(path).resolve()
        lang = lang or ("tirx" if path.suffix == ".py" else "cuda")
        return Candidate(workload, lang, path.read_text(), path=path, label=kw.pop("label", path.stem), **kw)

    def materialize(self) -> Path:
        """Return a file containing `source` (writes build/candidates/<hash>.<ext> if needed)."""
        if self.path and self.path.exists() and self.path.read_text() == self.source:
            return self.path
        digest = hashlib.sha1(self.source.encode()).hexdigest()[:12]
        p = BUILD / "candidates" / f"{self.label or 'cand'}_{digest}{'.py' if self.lang == 'tirx' else '.cu'}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.source)
        self.path = p
        return p


@dataclass
class ValidationContext:
    config: str = "C"                      # "A0" | "A" | "B" | "C" (informational for single validators)
    test_cases: list | None = None         # TestCases to run (None: validator's default)
    volta_shapes: list | None = None       # shapes for VOLTA (None: workload default)
    device: str = "auto"                   # "auto" | "gpu" | "cpu"
    time_budget_s: float = 600.0
    extra: dict = field(default_factory=dict)  # e.g. run_all (pipeline), classifier, budget, gate_shapes, gate_tools

    def with_(self, **kw) -> "ValidationContext":
        return replace(self, **kw)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "gpu" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@dataclass
class Finding:
    kind: str
    severity: str
    validator: str
    message: str
    evidence: dict = field(default_factory=dict)
    location: str | None = None

    def __post_init__(self):
        assert self.kind in FINDING_KINDS, self.kind
        assert self.severity in SEVERITIES, self.severity


@dataclass
class ValidationReport:
    validator: str
    passed: bool
    decided: bool
    findings: list[Finding] = field(default_factory=list)
    seconds: float = 0.0
    artifacts: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def build(validator: str, findings: list[Finding], decided: bool, t0: float, **artifacts) -> "ValidationReport":
        return ValidationReport(validator, not any(f.severity == "error" for f in findings), decided, findings,
                                time.perf_counter() - t0, artifacts)


class Validator(Protocol):
    name: str

    def validate(self, cand: Candidate, ctx: ValidationContext) -> ValidationReport: ...
