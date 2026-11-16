"""Load a PTX file and launch an `extern "C"` kernel from Python via the CUDA driver API.

Input : PTX (built by verify.ptx.compile_to_ptx(mode="gpu")), kernel name, grid/block, and the
        argument list [(value, ctypes type), ...] produced by ref.ops.OpSpec.kernel_args.
Output: the kernel runs on torch's current stream; errors raise CudaError with the driver's name.
Conditions: a CUDA GPU; torch initializes the primary context first (we then share it), so torch
tensors' data_ptr() values are valid kernel pointers. PTX is JIT-compiled by the driver (cached in
$CUDA_CACHE_PATH, set by env.sh).
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch
from cuda.bindings import driver as cu


class CudaError(RuntimeError):
    pass


def _check(result, what: str):
    err = result[0] if isinstance(result, tuple) else result
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        _, desc = cu.cuGetErrorString(err)
        raise CudaError(f"{what}: {name.decode() if name else err} ({desc.decode() if desc else ''})")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


class CudaFunction:
    def __init__(self, module: "CudaModule", fn, name: str):
        self._module, self._fn, self.name = module, fn, name   # keep module alive

    def __call__(self, grid, block, args, shared_bytes: int = 0, stream: int | None = None) -> None:
        objs = [ctype(value) for value, ctype in args]
        ptrs = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        _check(cu.cuLaunchKernel(self._fn, *grid, *block, shared_bytes, stream, ctypes.addressof(ptrs), 0),
               f"cuLaunchKernel({self.name})")


class CudaModule:
    def __init__(self, path: str | Path):
        torch.cuda.init()
        torch.empty(1, device="cuda")                    # make torch's primary context current
        data = Path(path).read_bytes()
        if str(path).endswith(".ptx"):
            data += b"\0"
        self.path = Path(path)
        self._mod = _check(cu.cuModuleLoadData(data), f"cuModuleLoadData({path})")

    def get(self, name: str) -> CudaFunction:
        fn = _check(cu.cuModuleGetFunction(self._mod, name.encode()), f"cuModuleGetFunction({name})")
        return CudaFunction(self, fn, name)
