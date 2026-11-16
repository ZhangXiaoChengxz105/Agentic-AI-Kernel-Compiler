"""Building block 1: compile a CUDA source file to PTX.

Input : a .cu file whose kernels are `extern "C" __global__` and use EXP(x) from workloads/common/common.cuh.
Output: PtxBuild (path to the .ptx file, kernel entry names found in it, the exact nvcc command, time).

Two modes:
  mode="volta": -DVOLTA (EXP -> call __symexpf), NO -lineinfo. VOLTA's lowering drops a label that
                is directly followed by a `.loc` directive ("Undefined label $L__BB0_21"), so
                line info must stay off for PTX that VOLTA reads.
  mode="gpu"  : plain build (EXP -> expf), -lineinfo on (useful for profilers / sanitizers).
Conditions: nvcc on PATH (module cudatoolkit), target arch defaults to sm_80 (A100).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import INCLUDE_DIR

DEFAULT_INCLUDES = (INCLUDE_DIR,)
_ENTRY_RE = re.compile(r"^\s*\.(?:visible\s+)?\.entry\s+([A-Za-z_$][\w$]*)\s*\(", re.M)


class CompileError(RuntimeError):
    """nvcc failed; .stderr holds the compiler output (useful as agent feedback)."""

    def __init__(self, cmd: list[str], stderr: str):
        super().__init__(f"nvcc failed: {' '.join(cmd)}\n{stderr}")
        self.cmd = cmd
        self.stderr = stderr


@dataclass(frozen=True)
class PtxBuild:
    source: Path
    ptx: Path
    mode: str
    arch: str
    kernels: tuple[str, ...]
    cmd: tuple[str, ...]
    seconds: float
    warnings: str = field(default="")


def compile_to_ptx(
    source: str | Path,
    out_dir: str | Path,
    *,
    mode: str = "volta",
    arch: str = "sm_80",
    include_dirs: tuple[Path, ...] = DEFAULT_INCLUDES,
    defines: dict[str, str] | None = None,
    nvcc: str | None = None,
) -> PtxBuild:
    if mode not in ("volta", "gpu"):
        raise ValueError(f"mode must be 'volta' or 'gpu', got {mode!r}")
    source = Path(source).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvcc = nvcc or shutil.which("nvcc")
    if not nvcc:
        raise FileNotFoundError("nvcc not found on PATH (module load cudatoolkit)")

    ptx = out_dir / f"{source.stem}.{mode}.ptx"
    cmd = [nvcc, f"-arch={arch}", "-ptx", str(source), "-o", str(ptx)]
    cmd += [f"-I{d}" for d in include_dirs]
    if mode == "volta":
        cmd.append("-DVOLTA")
    else:
        cmd.append("-lineinfo")
    for k, v in (defines or {}).items():
        cmd.append(f"-D{k}={v}" if v is not None else f"-D{k}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise CompileError(cmd, proc.stderr)
    kernels = tuple(_ENTRY_RE.findall(ptx.read_text()))
    return PtxBuild(source, ptx, mode, arch, kernels, tuple(cmd), seconds, proc.stderr.strip())
