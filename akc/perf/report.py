"""PerfReport: the one result format for kernel speed (the counterpart of validation's ValidationReport).

Speed is the agent's objective, not a validator: there is no pass/fail, only "how fast, and how far from
what the hardware allows". It is measured only for candidates that passed validation, and identically in
configurations A/B/C (otherwise the comparison of validation configurations would not be fair).

analyze(workload, shape, timing, torch_ms, copy_ms, spec, ...) -> PerfReport     pure arithmetic (CPU-testable)

Roofline terms used in the report
  bytes        minimum DRAM traffic of the op: each input read once, the output written once (OpSpec.bytes)
  flops        arithmetic of the op (OpSpec.flops)
  intensity    flops / bytes; below the GPU's ridge (A100: 12.5 FLOP/byte) the op is memory-bound
  achieved     bytes / time (GB/s) and flops / time (TFLOP/s); "effective" because they count the minimum
               traffic: a kernel that reads its input twice shows a lower effective bandwidth
  roofline_ms  max(bytes / peak bandwidth, flops / peak FP32): no kernel can be faster than this
  floor_ms     practical lower bound: a device copy of the same bytes (memory-bound ops), else roofline_ms
  headroom     time / floor_ms, e.g. 1.4 = "the floor is 1.4x faster than this kernel"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .hardware import GpuSpec

LATENCY_MS = 0.01        # below ~10 us, launch latency dominates and bandwidth numbers mean little


@dataclass
class PerfReport:
    workload: str
    label: str
    lang: str
    shape: dict
    measured: bool
    reason: str = ""                     # why not measured: no GPU, compile error, incorrect at this shape, ...
    gpu: str = ""
    time_ms: float | None = None         # median (L2 flushed before every launch)
    p10_ms: float | None = None
    p90_ms: float | None = None
    iters: int = 0
    torch_ms: float | None = None        # PyTorch's own implementation (OpSpec.torch_impl), same conditions
    copy_ms: float | None = None         # device copy of `bytes` bytes (memory-bound ops only)
    flops: float = 0.0
    bytes: float = 0.0
    intensity: float = 0.0
    bound: str = ""                      # "memory" | "compute" | "latency"
    achieved_gbps: float | None = None
    achieved_tflops: float | None = None
    pct_peak_bw: float | None = None
    pct_peak_flops: float | None = None
    roofline_ms: float | None = None
    floor_ms: float | None = None
    seconds: float = 0.0                 # wall time of the measurement itself
    spec: dict = field(default_factory=dict)

    @property
    def speedup_vs_torch(self) -> float | None:
        return self.torch_ms / self.time_ms if self.measured and self.torch_ms and self.time_ms else None

    @property
    def headroom(self) -> float | None:
        return self.time_ms / self.floor_ms if self.measured and self.floor_ms and self.time_ms else None

    def faster_than(self, other: "PerfReport | None") -> bool:
        """For keeping the best correct version: unmeasured reports never win."""
        if not self.measured or self.time_ms is None:
            return False
        return other is None or not other.measured or other.time_ms is None or self.time_ms < other.time_ms

    def to_dict(self) -> dict:
        return {**asdict(self), "speedup_vs_torch": self.speedup_vs_torch, "headroom": self.headroom}

    @staticmethod
    def from_dict(d: dict) -> "PerfReport":
        keys = PerfReport.__dataclass_fields__.keys()
        return PerfReport(**{k: v for k, v in d.items() if k in keys})

    @staticmethod
    def not_measured(workload: str, label: str, lang: str, shape: dict, reason: str, seconds: float = 0.0) -> "PerfReport":
        return PerfReport(workload, label, lang, dict(shape), False, reason, seconds=seconds)


def analyze(workload: str, shape: dict, timing: dict, spec: GpuSpec, *, label: str = "", lang: str = "cuda",
            torch_ms: float | None = None, copy_ms: float | None = None, seconds: float = 0.0) -> PerfReport:
    """Turn raw timings (backend.bench output) into a PerfReport with roofline numbers."""
    from ..task.reference import OPS
    op = OPS[workload]
    flops, nbytes, t = op.flops(shape), op.bytes(shape), timing["median_ms"]
    intensity = flops / nbytes
    gbps = nbytes / (t * 1e-3) / 1e9
    tflops = flops / (t * 1e-3) / 1e12
    roofline_ms = max(nbytes / (spec.dram_gbps * 1e9), flops / (spec.fp32_tflops * 1e12)) * 1e3
    if t < LATENCY_MS:
        bound = "latency"
    else:
        bound = "memory" if intensity < spec.ridge else "compute"
    floor_ms = copy_ms if (intensity < spec.ridge and copy_ms) else roofline_ms
    return PerfReport(
        workload, label, lang, dict(shape), True, gpu=spec.name, time_ms=t, p10_ms=timing.get("p10_ms"),
        p90_ms=timing.get("p90_ms"), iters=timing.get("iters", 0), torch_ms=torch_ms,
        copy_ms=copy_ms if intensity < spec.ridge else None, flops=flops, bytes=nbytes, intensity=intensity,
        bound=bound, achieved_gbps=gbps, achieved_tflops=tflops, pct_peak_bw=100 * gbps / spec.dram_gbps,
        pct_peak_flops=100 * tflops / spec.fp32_tflops, roofline_ms=roofline_ms, floor_ms=floor_ms,
        seconds=seconds, spec=spec.to_dict())
