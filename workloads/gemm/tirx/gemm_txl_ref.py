"""GEMM C = A @ B in tirx-lite for sm_80 (hand-written reference; NOT shown to the LLM).

make_kernel(m, n, k) -> txl.Kernel with A : float32 [m, k], B : float32 [k, n], C : float32 [m, n] (row-major).
Mirrors workloads/gemm/cuda/gemm_ref.cu: one thread per output element, no shared memory.
Launch: grid = (ceil(n / 16), ceil(m / 16)), block = 256 threads read as 16 x 16 (tirx-lite blocks are 1-D):
CTA (bx, by) owns output rows [16 by, 16 by + 16) x cols [16 bx, 16 bx + 16), like the CUDA tile2d launch.
"""
import tirx_kernels.tirx_lite as txl

TILE = 16


def make_kernel(m: int, n: int, k: int):
    @txl.kernel()
    def gemm_txl(A: txl.Tensor((m, k), "float32"), B: txl.Tensor((k, n), "float32"),
                 C: txl.Tensor((m, n), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=((n + TILE - 1) // TILE, (m + TILE - 1) // TILE),
                                                      block=TILE * TILE),
                         backend_config={"cuda": {"arch": "sm_80"}})
        bx, by = txl.cta_id()
        tid = txl.thread_id()
        row = by * TILE + tid // TILE
        col = bx * TILE + tid % TILE
        a = txl.local_scalar("float32")
        b = txl.local_scalar("float32")
        acc = txl.local_scalar("float32", init=txl.float32(0.0))
        with txl.if_(row < m), txl.then_():
            with txl.if_(col < n), txl.then_():
                with txl.serial(k) as kk:
                    txl.ptx.ld.global_.f32(a, A.ptr_to([row, kk]))
                    txl.ptx.ld.global_.f32(b, B.ptr_to([kk, col]))
                    txl.assign(acc, acc + a * b)
                txl.ptx.st.global_.f32(C.ptr_to([row, col]), acc)

    return gemm_txl
