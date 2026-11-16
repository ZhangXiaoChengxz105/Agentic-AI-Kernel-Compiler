"""Task 1.6: kernel timing on the GPU (VOLTA never runs kernels, so it cannot measure speed).

bench(fn) -> {"median_ms", "p10_ms", "p90_ms", "min_ms", "iters"}
  * warmup launches first (JIT, clocks, caches)
  * each timed launch is bracketed by CUDA events on the current stream
  * L2 is flushed before every timed launch by writing a 256 MiB buffer (A100 L2 = 40 MiB), so we
    measure cold-cache behaviour consistently instead of whatever the previous launch left behind
  * median over `iters` launches (robust to outliers)
"""
from __future__ import annotations

import statistics
from typing import Callable

import torch

FLUSH_BYTES = 256 << 20


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 50, flush_l2: bool = True) -> dict:
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda") if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    q = lambda p: times[min(len(times) - 1, int(p * len(times)))]  # noqa: E731
    return {"median_ms": statistics.median(times), "p10_ms": q(0.1), "p90_ms": q(0.9), "min_ms": times[0],
            "iters": iters}
