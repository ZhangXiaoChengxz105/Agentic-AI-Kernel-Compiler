"""Minimal CUDA driver-API launcher: the child process every compute-sanitizer tool wraps (compiler_checks/sanitizer.py).

    python akc/validation/compiler_checks/launcher.py spec.json out.json      (run BY PATH, not with -m)

Imports only the standard library, numpy and cuda.bindings: running it by path keeps the akc package (which imports
torch) out of the process, so no PyTorch start-up or PyTorch kernels run under the tools.

spec.json: {"units": [{"label", "cubin", "entry", "grid", "block", "smem",
                       "args": [{"name", "npy", "output": bool, "save": path (outputs)}]  (kernel-parameter order)}],
            "init_outputs": bool,    # copy the .npy (sentinel) into output buffers; False under initcheck
            "save_outputs": bool}    # copy outputs back and np.save them to args[i]["save"]
Every tensor is its own exact-size cuMemAlloc. Units run in order in one CUDA context; after a fault that kills the
context, the remaining units are marked "skipped". out.json: {"units": [{label, status, message?, kernel_fault?,
launch_ms?}], "error": setup error | None}; rewritten after every unit.
"""
import ctypes
import json
import sys
import time
from pathlib import Path

KERNEL_FAULTS = ("ILLEGAL_ADDRESS", "LAUNCH_FAILED", "MISALIGNED_ADDRESS", "ILLEGAL_INSTRUCTION", "ASSERT",
                 "LAUNCH_TIMEOUT", "LAUNCH_OUT_OF_RESOURCES", "HARDWARE_STACK_ERROR", "INVALID_PC",
                 "INVALID_ADDRESS_SPACE", "ILLEGAL_STATE")


def main(spec_path: str, out_path: str) -> int:
    import numpy as np
    from cuda.bindings import driver as cu

    def ck(res, what):
        err = res[0] if isinstance(res, tuple) else res
        if err != cu.CUresult.CUDA_SUCCESS:
            _, name = cu.cuGetErrorName(err)
            raise RuntimeError(f"{what}: {name.decode() if name else err}")
        return res[1] if isinstance(res, tuple) and len(res) > 1 else None

    spec = json.loads(Path(spec_path).read_text())
    res = {"units": [], "error": None}

    def save():
        Path(out_path).write_text(json.dumps(res))

    try:
        ck(cu.cuInit(0), "cuInit")
        dev = ck(cu.cuDeviceGet(0), "cuDeviceGet")
        ctx = ck(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
        ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    except Exception as e:  # noqa: BLE001
        res["error"] = f"setup: {e}"
        save()
        return 0
    dead = None
    for u in spec["units"]:
        rec = {"label": u["label"], "status": "ok"}
        if dead:
            res["units"].append({**rec, "status": "skipped", "message": dead})
            continue
        allocs, mod, stage = [], None, "load"
        try:
            mod = ck(cu.cuModuleLoadData(Path(u["cubin"]).read_bytes()), "cuModuleLoadData")
            fn = ck(cu.cuModuleGetFunction(mod, u["entry"].encode()), "cuModuleGetFunction")
            smem = int(u.get("smem") or 0)
            if smem > 48 * 1024:
                ck(cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         smem), "cuFuncSetAttribute")
            stage = "alloc"
            ptrs, outs = [], []
            for a in u["args"]:
                arr = np.ascontiguousarray(np.load(a["npy"]))
                p = ck(cu.cuMemAlloc(arr.nbytes), "cuMemAlloc")
                allocs.append(p)
                if not a.get("output") or spec.get("init_outputs", True):
                    ck(cu.cuMemcpyHtoD(p, arr.ctypes.data, arr.nbytes), "cuMemcpyHtoD")
                ptrs.append(ctypes.c_void_p(int(p)))
                if a.get("output"):
                    outs.append((a, p, arr))
            argv = (ctypes.c_void_p * len(ptrs))(*[ctypes.addressof(o) for o in ptrs])
            grid = (list(u["grid"]) + [1, 1, 1])[:3]
            block = (list(u["block"]) + [1, 1, 1])[:3]
            stage = "launch"
            t0 = time.perf_counter()
            ck(cu.cuLaunchKernel(fn, *grid, *block, smem, 0, ctypes.addressof(argv), 0), "cuLaunchKernel")
            ck(cu.cuCtxSynchronize(), "cuCtxSynchronize")
            rec["launch_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
            stage = "copy"
            if spec.get("save_outputs"):
                for a, p, arr in outs:
                    got = np.empty_like(arr)
                    ck(cu.cuMemcpyDtoH(got.ctypes.data, p, got.nbytes), "cuMemcpyDtoH")
                    np.save(a["save"], got)
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:300]
            rec.update(status="cuda_error" if stage in ("launch", "copy") else f"{stage}_error", message=msg,
                       kernel_fault=any(k in msg for k in KERNEL_FAULTS))
            alive = cu.cuCtxSynchronize()[0] == cu.CUresult.CUDA_SUCCESS    # memcheck may stop only the kernel
            dead = None if alive else msg
        finally:
            for p in allocs:
                cu.cuMemFree(p)
            if mod is not None:
                cu.cuModuleUnload(mod)
        res["units"].append(rec)
        print(f"@@ {u['label']}: {rec['status']}", flush=True)
        save()
    cu.cuDevicePrimaryCtxRelease(dev)
    save()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
