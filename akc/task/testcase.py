"""A single test input description, shared by the fixed suite (Task 1.5) and Task 2's generator.

TestCase(op, shape, dist, seed, repeat, tag)
  op     : "softmax" | "layernorm" | "gemm"
  shape  : op-specific sizes, e.g. {"rows": 64, "N": 1000} or {"M": 12, "N": 12, "K": 40}
  dist   : value distribution name from akc.task.inputs.DISTS
  seed   : RNG seed -> identical data on every run/machine
  repeat : run the kernel this many times; differing outputs => nondeterminism (race symptom)
  tag    : why this test exists (e.g. "fixed", "tiling: N = BLOCK+1"), shown in feedback
  misalign: shift every tensor pointer by this many elements (breaks 16-byte alignment that
           vectorized float4 loads/stores assume); 0 = normal allocation
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TestCase:
    op: str
    shape: dict = field(hash=False)
    dist: str = "normal"
    seed: int = 0
    repeat: int = 1
    tag: str = ""
    misalign: int = 0

    def key(self) -> str:
        dims = ",".join(f"{k}={v}" for k, v in self.shape.items())
        mis = f" misalign={self.misalign}" if self.misalign else ""
        return f"{self.op}[{dims}] dist={self.dist} seed={self.seed} x{self.repeat}{mis}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TestCase":
        return TestCase(d["op"], dict(d["shape"]), d.get("dist", "normal"), int(d.get("seed", 0)),
                        int(d.get("repeat", 1)), d.get("tag", ""), int(d.get("misalign", 0)))
