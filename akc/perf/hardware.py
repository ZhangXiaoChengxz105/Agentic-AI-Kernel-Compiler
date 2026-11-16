"""Peak numbers of the GPUs we may run on (datasheet values), used as the roofline ceilings.

GpuSpec.dram_gbps     peak DRAM (HBM) bandwidth, GB/s (1 GB = 1e9 bytes)
GpuSpec.fp32_tflops   peak FP32 FMA throughput (CUDA cores, not tensor cores), TFLOP/s
GpuSpec.ridge         FLOP per byte where a kernel stops being memory-bound and becomes compute-bound
Our GEMM tolerance needs full fp32 accuracy, so the FP32 (non-tensor-core) peak is the right compute ceiling.
Datasheet peaks are not reachable in practice: a plain device-to-device copy reaches ~80-90% of the DRAM
peak, which is why perf.measure also times a copy of the same number of bytes (the practical floor).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    dram_gbps: float
    fp32_tflops: float
    sms: int
    l2_mb: float
    known: bool = True                 # False: name not in the table, A100-40GB numbers used as a stand-in

    @property
    def ridge(self) -> float:
        return self.fp32_tflops * 1e12 / (self.dram_gbps * 1e9)

    def to_dict(self) -> dict:
        return {**asdict(self), "ridge_flop_per_byte": round(self.ridge, 2)}


# NVIDIA A100 datasheet: 40GB SXM4 1,555 GB/s; 80GB SXM4 2,039 GB/s; FP32 19.5 TFLOP/s; 108 SMs; 40 MB L2.
SPECS = {
    "A100-SXM4-40GB": GpuSpec("A100-SXM4-40GB", 1555.0, 19.5, 108, 40.0),
    "A100-SXM4-80GB": GpuSpec("A100-SXM4-80GB", 2039.0, 19.5, 108, 40.0),
}
DEFAULT = "A100-SXM4-40GB"             # NCSA Delta gpuA100x4 nodes


def spec_for(device_name: str | None) -> GpuSpec:
    """Match a torch device name such as 'NVIDIA A100-SXM4-40GB'; unknown GPUs get the A100-40GB numbers."""
    for key, spec in SPECS.items():
        if device_name and key in device_name:
            return spec
    d = SPECS[DEFAULT]
    return GpuSpec(device_name or "unknown", d.dram_gbps, d.fp32_tflops, d.sms, d.l2_mb, known=False)
